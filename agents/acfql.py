import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value

class ACFQLAgent(flax.struct.PyTreeNode):
    """Flow Q-learning (FQL) agent with action chunking. 
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()
    # Decode a = clip(mean + (z * pca_bound * std) @ components, -1, 1). z is not clipped.
    pca_mean: Any = None
    pca_components: Any = None
    pca_scale: Any = None

    def critic_loss(self, batch, grad_params, rng):
        """Compute the FQL critic loss."""

        batch_actions = self._flat_actions(batch["actions"])
        if self.config["pca_latent"]:
            batch_actions = self._chunk_to_latent(batch_actions)

        # TD loss
        rng, sample_rng = jax.random.split(rng)
        next_obs = batch['next_observations'][..., -1, :]
        if self.config["pca_latent"]:
            # Bootstrap in latent space. sample_actions decodes to env chunks.
            next_actions = self._onestep_latent(next_obs, sample_rng)
        else:
            next_actions = self.sample_actions(next_obs, rng=sample_rng)

        next_qs = self.network.select(f'target_critic')(batch['next_observations'][..., -1, :], actions=next_actions)
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)
        
        target_q = batch['rewards'][..., -1] + \
            (self.config['discount'] ** self.config["horizon_length"]) * batch['masks'][..., -1] * next_q

        q = self.network.select('critic')(batch['observations'], actions=batch_actions, params=grad_params)
        
        critic_loss = (jnp.square(q - target_q) * batch['valid'][..., -1]).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss."""
        batch_actions = self._flat_actions(batch["actions"])
        if self.config["pca_latent"]:
            batch_actions = self._chunk_to_latent(batch_actions)
        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)

        # only bc on the valid chunk indices
        if self.config["pca_latent"]:
            # One latent per chunk. Drop chunks that cross a terminal.
            bc_flow_loss = jnp.mean(jnp.mean((pred - vel) ** 2, axis=-1) * batch["valid"][..., -1])
        elif self.config["action_chunking"]:
            bc_flow_loss = jnp.mean(
                jnp.reshape(
                    (pred - vel) ** 2, 
                    (batch_size, self.config["horizon_length"], self.config["action_dim"]) 
                ) * batch["valid"][..., None]
            )
        else:
            bc_flow_loss = jnp.mean(jnp.square(pred - vel))

        if self.config["actor_type"] == "distill-ddpg":
            # Distillation loss. Shared noise so the one-step actor matches this flow sample.
            rng, noise_rng = jax.random.split(rng)
            noises = jax.random.normal(noise_rng, (batch_size, action_dim))
            target_flow_actions = self.compute_flow_actions(batch['observations'], noises=noises)
            actor_actions = self.network.select('actor_onestep_flow')(
                batch['observations'], noises, params=grad_params
            )
            distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)
            # pca_latent keeps actor_actions in latent space. Clip only env actions.
            if not self.config["pca_latent"]:
                actor_actions = jnp.clip(actor_actions, -1, 1)

            qs = self.network.select(f'critic')(batch['observations'], actions=actor_actions)
            q = jnp.mean(qs, axis=0)
            q_loss = -q.mean()
        else:
            distill_loss = jnp.zeros(())
            q_loss = jnp.zeros(())

        # Total loss.
        actor_loss = bc_flow_loss + self.config['alpha'] * distill_loss + q_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': distill_loss,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @staticmethod
    def _update(agent, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, 'critic')
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)
    
    @jax.jit
    def batch_update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        # update_size = batch["observations"].shape[0]
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)
    
    @jax.jit
    def sample_actions(
        self,
        observations,
        rng=None,
    ):
        
        if self.config["actor_type"] == "distill-ddpg":
            batch_shape = observations.shape[: -len(self.config['ob_dims'])]
            if self.config["pca_latent"]:
                actions = self._latent_to_chunk(self._onestep_latent(observations, rng))
            else:
                noises = jax.random.normal(
                    rng,
                    (
                        *batch_shape,
                        self.config['action_dim'] * \
                            (self.config['horizon_length'] if self.config["action_chunking"] else 1),
                    ),
                )
                actions = self.network.select('actor_onestep_flow')(observations, noises)
            actions = jnp.clip(actions, -1, 1)

        elif self.config["actor_type"] == "best-of-n":
            action_dim = self.config['action_dim'] * \
                        (self.config['horizon_length'] if self.config["action_chunking"] else 1)
            noises = jax.random.normal(
                rng,
                (
                    *observations.shape[: -len(self.config['ob_dims'])],  # batch_size
                    self.config["actor_num_samples"], action_dim
                ),
            )
            observations = jnp.repeat(observations[..., None, :], self.config["actor_num_samples"], axis=-2)
            actions = self.compute_flow_actions(observations, noises)
            actions = jnp.clip(actions, -1, 1)
            if self.config["q_agg"] == "mean":
                q = self.network.select("critic")(observations, actions).mean(axis=0)
            else:
                q = self.network.select("critic")(observations, actions).min(axis=0)
            indices = jnp.argmax(q, axis=-1)

            bshape = indices.shape
            indices = indices.reshape(-1)
            bsize = len(indices)
            actions = jnp.reshape(actions, (-1, self.config["actor_num_samples"], action_dim))[jnp.arange(bsize), indices, :].reshape(
                bshape + (action_dim,))

        return actions

    @jax.jit
    def compute_flow_actions(
        self,
        observations,
        noises,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_bc_flow_encoder')(observations)
        actions = noises
        # Euler method.
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_bc_flow')(observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        if not self.config["pca_latent"]:
            actions = jnp.clip(actions, -1, 1)
        return actions

    def _flat_actions(self, actions):
        if self.config["action_chunking"]:
            return jnp.reshape(actions, (actions.shape[0], -1))
        return actions[..., 0, :]

    def _latent_scale(self):
        """Action-space width of one latent unit: pca_bound standard deviations."""
        return self.pca_scale * self.config["pca_bound"]

    def _chunk_to_latent(self, flat_actions):
        """Project a flat action chunk to an unclipped PCA latent."""
        return (flat_actions - self.pca_mean) @ self.pca_components.T / self._latent_scale()

    def _latent_to_chunk(self, latent):
        """Decode a PCA latent and clip the chunk to the original action box."""
        flat = self.pca_mean + (latent * self._latent_scale()) @ self.pca_components
        return jnp.clip(flat, -1.0, 1.0)

    def _onestep_latent(self, observations, rng):
        batch_shape = observations.shape[: -len(self.config["ob_dims"])]
        noises = jax.random.normal(rng, (*batch_shape, self.config["pca_k"]))
        return self.network.select("actor_onestep_flow")(observations, noises)

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        if config["pca_latent"]:
            if not config["pca_k"]:
                raise ValueError("pca_latent requires pca_k")
            # Critic, flow, and one-step actor all take PCA latents.
            full_actions = jnp.zeros((config["pca_k"],), dtype=jnp.float32)
        elif config["action_chunking"]:
            full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
        else:
            full_actions = ex_actions
        network = _build_train_state(config, ex_observations, full_actions, init_rng)

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def _build_train_state(config, ex_observations, full_actions, init_rng):
    """Build critic, flow, and one-step actor."""
    ex_times = full_actions[..., :1]
    full_action_dim = full_actions.shape[-1]

    encoders = dict()
    if config['encoder'] is not None:
        encoder_module = encoder_modules[config['encoder']]
        encoders['critic'] = encoder_module()
        encoders['actor_bc_flow'] = encoder_module()
        encoders['actor_onestep_flow'] = encoder_module()

    critic_def = Value(
        hidden_dims=config['value_hidden_dims'],
        layer_norm=config['layer_norm'],
        num_ensembles=config['num_qs'],
        encoder=encoders.get('critic'),
    )
    actor_bc_flow_def = ActorVectorField(
        hidden_dims=config['actor_hidden_dims'],
        action_dim=full_action_dim,
        layer_norm=config['actor_layer_norm'],
        encoder=encoders.get('actor_bc_flow'),
        use_fourier_features=config["use_fourier_features"],
        fourier_feature_dim=config["fourier_feature_dim"],
    )
    actor_onestep_flow_def = ActorVectorField(
        hidden_dims=config['actor_hidden_dims'],
        action_dim=full_action_dim,
        layer_norm=config['actor_layer_norm'],
        encoder=encoders.get('actor_onestep_flow'),
    )

    network_info = dict(
        actor_bc_flow=(actor_bc_flow_def, (ex_observations, full_actions, ex_times)),
        actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, full_actions)),
        critic=(critic_def, (ex_observations, full_actions)),
        target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
    )
    if encoders.get('actor_bc_flow') is not None:
        network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_observations,))
    networks = {k: v[0] for k, v in network_info.items()}
    network_args = {k: v[1] for k, v in network_info.items()}

    network_def = ModuleDict(networks)
    if config["weight_decay"] > 0.:
        network_tx = optax.adamw(learning_rate=config['lr'], weight_decay=config["weight_decay"])
    else:
        network_tx = optax.adam(learning_rate=config['lr'])
    network_params = network_def.init(init_rng, **network_args)['params']
    network = TrainState.create(network_def, network_params, tx=network_tx)
    params = network.params
    params[f'modules_target_critic'] = params[f'modules_critic']
    return network


def fit_chunk_pca(dataset, horizon, k=None, variance=None, seed=0, max_chunks=200_000):
    """PCA on offline action chunks that stay inside one episode.

    k fixes the width. variance, in (0, 1], instead picks the smallest width whose
    cumulative explained variance reaches that fraction. Returns mean (D,),
    components (K, D), per-component std (K,), and the cumulative fraction.
    The decode is clip(mean + (pca_bound * std * z) @ components, -1, 1). z is not clipped.
    """
    actions = np.asarray(dataset['actions'])
    starts = []
    for ep_start, ep_end in zip(dataset.initial_locs, dataset.terminal_locs):
        last = int(ep_end) - int(horizon) + 1
        if last >= int(ep_start):
            starts.append(np.arange(int(ep_start), last + 1))
    if not starts:
        raise ValueError(f"no episode is long enough for horizon {horizon}")
    starts = np.concatenate(starts)
    rng = np.random.default_rng(seed)
    if len(starts) > max_chunks:
        starts = rng.choice(starts, size=max_chunks, replace=False)

    idx = starts[:, None] + np.arange(horizon)[None, :]
    chunks = actions[idx].reshape(len(starts), -1).astype(np.float64)
    mean = chunks.mean(axis=0, keepdims=True)
    _, singular, vt = np.linalg.svd(chunks - mean, full_matrices=False)
    denom = max(len(starts) - 1, 1)
    variance_all = (singular ** 2) / denom
    total = max(float(variance_all.sum()), 1e-12)
    cumsum = np.cumsum(variance_all) / total
    if variance is not None:
        target = float(variance)
        if not 0 < target <= 1:
            raise ValueError(f"PCA variance target must be in (0, 1], got {target}")
        k_keep = int(np.searchsorted(cumsum, target) + 1)
        k_keep = min(k_keep, len(cumsum))
    else:
        if k is None or int(k) <= 0:
            raise ValueError("fit_chunk_pca needs k > 0 or a variance target")
        k_keep = min(int(k), len(cumsum))
    explained = float(cumsum[k_keep - 1])
    ratios = variance_all[:k_keep] / total
    shown = min(8, k_keep)
    print(
        f"chunk PCA k={k_keep} cumsum={explained:.3f}"
        + (f" target={float(variance):.3f}" if variance is not None else "")
        + f" n={len(starts)} dim={chunks.shape[1]} top="
        + ", ".join(f"{r:.3f}" for r in ratios[:shown])
    )
    scale = np.maximum(np.sqrt(variance_all[:k_keep]), 1e-6).astype(np.float32)
    return mean.reshape(-1).astype(np.float32), vt[:k_keep].astype(np.float32), scale, explained


def get_config():

    config = ml_collections.ConfigDict(
        dict(
            agent_name='acfql',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            q_agg='mean',  # Aggregation method for target Q values.
            alpha=100.0,  # BC coefficient (need to be tuned for each environment).
            num_qs=2, # critic ensemble size
            flow_steps=10,  # Number of flow steps.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            horizon_length=ml_collections.config_dict.placeholder(int), # will be set
            action_chunking=True,  # False means n-step return
            actor_type="distill-ddpg",
            actor_num_samples=32,  # for actor_type="best-of-n" only
            pca_k=0,  # PCA latent width. 0 keeps the full chunk.
            pca_latent=False,  # True: critic, flow, and one-step actor train in the PCA latent.
            pca_bound=1.0,  # One latent unit is this many standard deviations. Latents are not clipped.
            use_fourier_features=False,
            fourier_feature_dim=64,
            weight_decay=0.,
        )
    )
    return config
