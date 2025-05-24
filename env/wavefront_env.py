# WavefrontEnv: Reinforcement Learning environment for wavefront shaping

import numpy as np

class WavefrontEnv:
    def __init__(self, slm_dim1=64, slm_dim2=64, eng_size=1,
                 num_pix_per_block=32, alpha=0.3, noise_sigma=0.05):
        # geometry -----------------------------------------------------------
        self.slm_dim1, self.slm_dim2 = slm_dim1, slm_dim2
        self.n_pix   = slm_dim1 * slm_dim2            # 4096
        self.eng_size = eng_size                      # up-sampling factor

        # block grid ---------------------------------------------------------
        self.blocks = self._make_blocks(num_pix_per_block)
        self.num_blocks = len(self.blocks)            # 128 for 32-pix blocks

        # RL bookkeeping -----------------------------------------------------
        self.state_dim    = self.num_blocks + 1       # 128 bits + intensity
        self.action_space = self.num_blocks
        self.alpha  = alpha
        self.sigma  = noise_sigma                     # measurement noise σ

        self.reset()

    # ------------------------------------------------------------------ utils
    def _make_blocks(self, p_per_block):
        """Return list of numpy arrays – each array holds pixel indices of one block."""
        idx = np.arange(self.n_pix)
        return [idx[k : k + p_per_block]
                for k in range(0, self.n_pix, p_per_block)]

    def _blocks_to_pixels(self):
        """Convert block-level mask → 2-D pixel mask (0/1)."""
        pixel = np.zeros(self.n_pix, dtype=np.float32)
        for bid, bit in enumerate(self.block_mask):
            if bit:                                     # block is ON
                pixel[self.blocks[bid]] = 1.0
        return pixel.reshape(self.slm_dim1, self.slm_dim2)

    # ------------------------------------------------------------- RL methods
    def reset(self):
        self.block_mask = np.ones(self.num_blocks, dtype=np.float32)   # all blocks ON
        self.phi        = np.random.rand(self.slm_dim1, self.slm_dim2) # random phase
        self.I_max  = 0.0
        self.I_prev = self._intensity()
        self.I0_mean = self.I_prev
        return self._state()

    def step(self, action: int):
        # flip chosen block bit
        self.block_mask[action] = 1.0 - self.block_mask[action]

        I_t   = self._intensity()
        delta = (I_t - self.I_prev) / self.I0_mean
        bonus = max(0.0, I_t - self.I_max) / self.I0_mean
        reward = delta + self.alpha * bonus - 1e-4      # small time penalty λ

        # bookkeeping
        self.I_prev = I_t
        self.I_max  = max(self.I_max, I_t)

        return self._state(), reward

    # ---------------------------------------------------------------- optics
    def _intensity(self):
        mask2d = self._blocks_to_pixels()
        field  = np.exp(1j * 2 * np.pi * self.phi) * mask2d
        spec   = np.fft.fftshift(np.fft.fft2(field))
        I      = np.abs(spec[self.slm_dim1 // 2, self.slm_dim2 // 2])**2 / spec.size
        I     += self.sigma * np.random.randn()         # additive Gaussian noise
        return float(I)

    # ---------------------------------------------------------------- state
    def _state(self):
        norm_I = self.I_prev / self.I0_mean
        return np.concatenate([self.block_mask, [norm_I]]).astype(np.float32)