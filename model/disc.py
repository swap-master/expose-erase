import torch
import torch.nn as nn
from einops import rearrange, repeat


def joints_to_bones(input_joints, epsilon=1e-8):
    """
    input_joints: N C T J M
    j2b_matrix: (24 x 25)

    bones_norm: N M T B C
    bones_len: N M T B 1
    skeleton_center: N M T 1 C
    """
    j2b_matrix = torch.tensor([[ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                                0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.],
                                [ 1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                                0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                                [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,
                                0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                                [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                                0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                                [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                                0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.],
                                [ 0.,  0.,  1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                                0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                                [ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                                0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.],
                                [ 0.,  0.,  0.,  0.,  1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                                0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                                [ 0.,  0.,  0.,  0.,  0.,  1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                                0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                                [ 0.,  0.,  0.,  0.,  0.,  0.,  1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,
                                0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                                [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,
                                0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.],
                                [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,
                                0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.],
                                [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,
                                0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.],
                                [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1., -1.,  0.,  0.,  0.,  0.,
                                0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                                [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1., -1.,  0.,  0.,  0.,
                                0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                                [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1., -1.,  0.,  0.,
                                0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                                [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,
                                0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.],
                                [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,
                                0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.],
                                [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1., -1.,
                                0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                                [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,
                                -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                                [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                                1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                                [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                                0.,  0.,  1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                                [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                                0.,  0.,  0.,  1., -1.,  0.,  0.,  0.,  0.,  0.,  0.],
                                [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                                0.,  0.,  0.,  0.,  1., -1.,  0.,  0.,  0.,  0.,  0.]])


    n, c, t, j, m = input_joints.shape
    j2b_matrix = j2b_matrix.to(input_joints.device)
    #skeleton_center = rearrange(input_joints[:, :, :, 1:2, :], 'N C T J M -> N M T J C')
    tmp_joints = rearrange(input_joints, 'N C T J M -> (N M) T C J', N=n, C=c, T=t, J=j, M=m)
    tmp_j2b_matrix = rearrange(j2b_matrix, 'B J -> J B')
    bones = torch.matmul(tmp_joints, tmp_j2b_matrix.cuda())
    tmp_bones = rearrange(bones, 'N T C J -> N J T C')
    tmp_bones_len = torch.norm(tmp_bones, dim=-1, keepdim=True)
    tmp_bones_norm = tmp_bones / (tmp_bones_len + epsilon)
    out_bones_norm = rearrange(tmp_bones_norm, '(N M) J T C -> N M T J C', M=m)
    #out_bones_len = rearrange(tmp_bones_len, '(N M) J T C -> N M T J C', M=m)
    return out_bones_norm#, out_bones_len, skeleton_center


class DiscriminatorWithNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(72, 128),
            nn.LayerNorm(128),  # Batch Normalization for 1D input (features)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        out_bones_norm = joints_to_bones(x)
        out_bones_norm = rearrange(out_bones_norm, 'N M T B C ->(N M T) (B C)')
        return self.model(out_bones_norm)


def hinge_loss_discriminator(real_output, fake_output):
    """
    Hinge loss for the discriminator.

    Args:
        real_output (torch.Tensor): Output of the discriminator for real samples.
        fake_output (torch.Tensor): Output of the discriminator for fake samples.

    Returns:
        torch.Tensor: Discriminator loss.
    """
    real_loss = torch.mean(torch.relu(1. - real_output))
    fake_loss = torch.mean(torch.relu(1. + fake_output))
    discriminator_loss = real_loss + fake_loss
    return discriminator_loss

def hinge_loss_generator(fake_output):
    """
    Hinge loss for the generator.

    Args:
        fake_output (torch.Tensor): Output of the discriminator for fake samples.

    Returns:
        torch.Tensor: Generator loss.
    """
    generator_loss = -torch.mean(fake_output)
    return generator_loss
