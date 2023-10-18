import torch

class StraightThrougClamp(torch.autograd.Function):

    @staticmethod
    def forward(self, input, low=-0.5, high=0.5):
        v = torch.clamp(input, low, high)
        self.save_for_backward(input, torch.tensor([low], dtype=torch.float32).to(
            input.device), torch.tensor([high], dtype=torch.float32).to(input.device))
        return v

    @staticmethod
    def backward(self, grad_output):
        input, low, high = self.saved_variables
        grad_input = grad_output
        grad_input = torch.where((input > high) & (
            grad_input < 0), 0*grad_input, grad_input)
        grad_input = torch.where((input < low) & (
            grad_input > 0), 0*grad_input, grad_input)
        return grad_input, None, None
