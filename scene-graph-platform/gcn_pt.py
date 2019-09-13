import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
class Graph(nn.Module):
    def __init__(self, layer_num, num_cls, hidden_dim, knowledge):
        super(Graph, self).__init__()
        self.layer_num = layer_num
        self.num_cls = num_cls
        self.hidden_dim = hidden_dim

        if knowledge is None:
            self.knowledge = np.ones((self.num_cls, self.num_cls)).astype(np.float32) / self.num_cls
        else:
            self.knowledge = knowledge

        self.knowledge = Variable(torch.from_numpy(self.knowledge), requires_grad=True).cuda()

        self.fc_eq3_w = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_eq3_u = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq4_w = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_eq4_u = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq5_w = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_eq5_u = nn.Linear(hidden_dim, hidden_dim)

    #shape of input [num_nodes, hidden_dim]
    #input is a torch tensor
    def forward(self, input):
        num_nodes = input.shape[0]
        hidden = input.repeat(1, self.num_cls).view(num_nodes, self.num_cls, -1)
        for i in self.layer_num:
            hidden_sum = torch.sum(hidden, 0)
            av = torch.cat([torch.cat([self.knowledge.transpose(0, 1) @ (hidden_sum - hidden_i) for hidden_i in hidden], 0),
                            torch.cat([self.knowledge @ (hidden_sum - hidden_i) for hidden_i in hidden], 0)], 1)
            hidden = hidden.view(num_nodes * self.num_cls, -1)
            zv = torch.sigmoid(self.fc_eq3_w(av) + self.fc_eq3_u(hidden))

            # eq(4)
            rv = torch.sigmoid(self.fc_eq4_w(av) + self.fc_eq3_u(hidden))

            # eq(5)
            hv = torch.tanh(self.fc_eq5_w(av) + self.fc_eq5_u(rv * hidden))

            hidden = (1 - zv) * hidden + zv * hv
            hidden = hidden.view(num_nodes, self.num_cls, -1)
        return hidden