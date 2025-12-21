import math

import torch
import torch.distributions as td


class MSE(td.Normal):
    def __init__(self, loc, validate_args=None):
        super().__init__(loc, 1.0, validate_args=validate_args)

    @property
    def mode(self):
        return self.mean

    def sample(self, sample_shape=torch.Size()):
        return self.rsample(sample_shape)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return -((value - self.loc) ** 2) / 2


class TruncatedStandardNormal(td.Distribution):
    arg_constraints = {"a": td.constraints.real, "b": td.constraints.real}
    has_rsample = True

    def __init__(self, a, b, validate_args=None):
        self.a, self.b = td.utils.broadcast_all(a, b)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            batch_shape = torch.Size()
        else:
            batch_shape = self.a.size()
        super().__init__(batch_shape, validate_args=validate_args)
        if self.a.dtype != self.b.dtype:
            raise ValueError("Truncation bounds types are different")
        eps = torch.finfo(self.a.dtype).eps
        if torch.any((self.a >= self.b).view(-1)):
            raise ValueError("Incorrect truncation range")
        self._dtype_min_gt_0 = eps
        self._dtype_max_lt_1 = 1 - eps
        self._little_phi_a = self._little_phi(self.a)
        self._little_phi_b = self._little_phi(self.b)
        self._big_phi_a = self._big_phi(self.a)
        self._big_phi_b = self._big_phi(self.b)
        self._Z = (self._big_phi_b - self._big_phi_a).clamp_min(eps)
        self._log_Z = self._Z.log()
        self._lpbb_m_lpaa_d_Z = (self._little_phi_b * self.b - self._little_phi_a * self.a) / self._Z
        self._mean = -(self._little_phi_b - self._little_phi_a) / self._Z
        self._mode = torch.clamp(torch.zeros_like(self.a), self.a, self.b)
        self._variance = 1 - self._lpbb_m_lpaa_d_Z - ((self._little_phi_b - self._little_phi_a) / self._Z) ** 2
        two_pi_e = torch.tensor(2 * math.pi * math.e, device=self.a.device, dtype=self.a.dtype)
        self._entropy = 0.5 * torch.log(two_pi_e) + self._log_Z - 0.5 * self._lpbb_m_lpaa_d_Z

    @staticmethod
    def _little_phi(x):
        two_pi = torch.tensor(2 * math.pi, device=x.device, dtype=x.dtype)
        return (-(x**2) * 0.5).exp() / torch.sqrt(two_pi)

    @staticmethod
    def _big_phi(x):
        sqrt_two = torch.tensor(math.sqrt(2.0), device=x.device, dtype=x.dtype)
        return 0.5 * (1 + (x / sqrt_two).erf())

    @staticmethod
    def _inv_big_phi(x):
        sqrt_two = torch.tensor(math.sqrt(2.0), device=x.device, dtype=x.dtype)
        return sqrt_two * (2 * x - 1).erfinv()

    @property
    def support(self):
        return td.constraints.interval(self.a, self.b)

    @property
    def mean(self):
        return self._mean

    @property
    def mode(self):
        return self._mode

    @property
    def variance(self):
        return self._variance

    def entropy(self):
        return self._entropy

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return ((self._big_phi(value) - self._big_phi_a) / self._Z).clamp(0, 1)

    def icdf(self, value):
        return self._inv_big_phi(self._big_phi_a + value * self._Z)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        two_pi = torch.tensor(2 * math.pi, device=value.device, dtype=value.dtype)
        return -self._log_Z - 0.5 * (value**2) - 0.5 * torch.log(two_pi)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        uniform = torch.empty(shape, device=self.a.device).uniform_(self._dtype_min_gt_0, self._dtype_max_lt_1)
        return self.icdf(uniform)


class TruncatedNormal(TruncatedStandardNormal):
    has_rsample = True

    def __init__(self, loc, scale, low, high, validate_args=None):
        self.loc, self.scale, low, high = td.utils.broadcast_all(loc, scale, low, high)
        standardized_low = (low - self.loc) / self.scale
        standardized_high = (high - self.loc) / self.scale
        super().__init__(standardized_low, standardized_high, validate_args=validate_args)
        self._log_scale = self.scale.log()
        self._mean = self._mean * self.scale + self.loc
        self._mode = torch.clamp(self.loc, low, high)
        self._variance = self._variance * self.scale**2
        self._entropy += self._log_scale

    def cdf(self, value):
        return super().cdf((value - self.loc) / self.scale)

    def icdf(self, value):
        return super().icdf(value) * self.scale + self.loc

    def log_prob(self, value):
        return super().log_prob((value - self.loc) / self.scale) - self._log_scale


class TruncNormalDist(TruncatedNormal):
    def __init__(self, loc, scale, low, high, clip=1e-6, mult=1):
        super().__init__(loc, scale, low, high)
        self._clip = clip
        self._mult = mult
        self.low = low
        self.high = high

    def sample(self, *args, **kwargs):
        event = super().rsample(*args, **kwargs)
        if self._clip:
            clipped = torch.clamp(event, self.low + self._clip, self.high - self._clip)
            event = event - event.detach() + clipped.detach()
        if self._mult:
            event *= self._mult
        return event
