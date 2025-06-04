# WavefrontEnv: Reinforcement Learning environment for wavefront shaping

import numpy as np

class WavefrontEnv:
    def __init__(self, slm_dim1=64, slm_dim2=64, eng_size=1,
                 num_pix_per_block=32, alpha=0.3, noise_sigma=0.05, k=0.45, phi=None):
        # geometry
        self.slm_dim1, self.slm_dim2 = slm_dim1, slm_dim2
        self.n_pix   = slm_dim1 * slm_dim2            
        self.eng_size = eng_size                      

        # phase mask
        if phi is None:
            self.phi = np.random.rand(slm_dim1, slm_dim2)
        else:
            assert phi.shape == (slm_dim1, slm_dim2), "Phase mask shape mismatch."
            self.phi = phi

        # block grid
        self.blocks = self._make_blocks(num_pix_per_block)
        self.num_blocks = len(self.blocks)            

        # RL bookkeeping
        self.state_dim    = self.num_blocks           
        self.action_space = self.num_blocks
        self.alpha  = alpha
        self.sigma  = noise_sigma                     
        self.I0_mean = 0.0
        self.I_max  = 0.0
        self.I_t   = 0.0

        self.k = k

        self.best_mask = None

        self.set_I0_mean()

        self.reset()

    #utils
    def _make_blocks(self, p_per_block):
        """Return list of numpy arrays, each array holds pixel indices of one block."""
        idx = np.arange(self.n_pix)
        return [idx[k : k + p_per_block]
                for k in range(0, self.n_pix, p_per_block)]

    def _blocks_to_pixels(self):
        """Convert block-level mask → 2-D pixel mask."""
        pixel = np.zeros(self.n_pix, dtype=np.float32)
        for bid, bit in enumerate(self.block_mask):
            if bit:                                     
                pixel[self.blocks[bid]] = 1.0
        return pixel.reshape(self.slm_dim1, self.slm_dim2)

    def reset(self):
        if self.best_mask is not None:
            if np.random.rand() < 0.2:
                self.block_mask = self.best_mask.copy()

                # Flip a small number of bits randomly (e.g., 5 out of 128)
                flip_indices = np.random.choice(self.num_blocks, size=5, replace=False)
                for idx in flip_indices:
                    self.block_mask[idx] = 1.0 - self.block_mask[idx]
            else:
                self.block_mask = self.best_mask.copy()
        else:
            self.block_mask = np.random.choice([0.0, 1.0], size=self.num_blocks).astype(np.float32)
            
        self.I_prev = self._intensity()
        self.I_max  = self.I_prev
        return self._state()
    
    def tanh_reward(self, I_1):
        return np.tanh(self.k * I_1)

    def step(self, action):
        
        for a in action:
            self.block_mask[a] = 1.0 - self.block_mask[a]

        self.I_t = self._intensity()
        reward = self.tanh_reward(self.I_t)
         
        # Bookkeeping best mask
        if self.I_t > self.I_max:
            self.I_max = self.I_t
            self.best_mask = self.block_mask.copy()

        return self._state(), reward

    #optics
    def _intensity(self):
        mask2d = self._blocks_to_pixels()
        field  = np.exp(1j * 2 * np.pi * self.phi) * mask2d
        spec   = np.fft.fftshift(np.fft.fft2(field))
        I      = np.abs(spec[self.slm_dim1 // 2, self.slm_dim2 // 2])**2 / spec.size
        I     += self.sigma * np.random.randn()         # additive Gaussian noise
        return float(I)

    #state
    def _state(self):
        return self.block_mask.astype(np.float32)
    
    def set_I0_mean(self):
        for i in range(1000):
            self.block_mask = np.random.choice([0.0, 1.0], size=self.num_blocks).astype(np.float32)
            I = self._intensity()
            self.I0_mean += I
        self.I0_mean /= 1000.0

