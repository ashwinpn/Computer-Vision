import torch
import torch.nn
import torch.nn.functional as F
from tqdm import tqdm
import gc
from run_nerf_helpers import *
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import argparse
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument(dest="nerf_path", type=str, help="Path to NeRF file")
parser.add_argument("--t_depth", type=int, default=8, help="Depth of teacher NeRF")
parser.add_argument("--t_width", type=int, default=256, help="Width of teacher NeRF")
parser.add_argument("--s_depth", type=int, default=8, help="Depth of student NeRF")
parser.add_argument("--s_width", type=int, default=256, help="Width of student NeRF")
parser.add_argument("--s_skips", nargs='+', default=[4], type=int, help="Skip connections to be used in student")
parser.add_argument("--input_ch", type=int, default=63, help="Number of input channels, after positional enocding")
parser.add_argument("--input_ch_views", type=int, default=27, help="Number of input channels (views), after positional encoding")
parser.add_argument("--log_freq", type=int, default=10, help="Frequency to log statisitics during training")
parser.add_argument("--status_freq", type=int, default=1000, help="Frequency to output status during training")
parser.add_argument("--lr", type=float, default=5e-4, help="Initial learning rate for distillation")
parser.add_argument("--loss_thresh", type=float, default=.1, help="Active layers are done training when total loss is below this amount")
parser.add_argument("--max_epochs", type=int, default=200000, help="Number of epochs to train for")
parser.add_argument("--layer_queue", type=str, default="0,0|1,1|2,2|3,3|4,4|5,5|6,6|7,7|8,8|9,9|O,O", help="Layers to be compared during distillation")
parser.add_argument("--plot_file", type=str, default="./plots/layer_{}_.png", help="Path to save plots to, include {} for layer number")

args = parser.parse_args()

tmp = []
for pair in args.layer_queue.split('|'):
    s,t = pair.split(',')
    try:
        tmp.append((int(s),int(t)))
    except ValueError:
        tmp.append((s,t))
args.layer_queue = deque(tmp)
del tmp

print("Arguments received:")
for arg in args.__dict__:
    print(arg, '=', getattr(args, arg))
print("")

# Load pretrained "teacher" NeRF models
saved = torch.load(args.nerf_path)
teacher_model = NeRF(D=args.t_depth, W=args.t_width, input_ch=args.input_ch, input_ch_views=args.input_ch_views, use_viewdirs=True)
teacher_model.load_state_dict(saved['network_fn_state_dict'])
teacher_model.eval()
teacher_model_fine = NeRF(D=args.t_depth, W=args.t_width, input_ch=args.input_ch, input_ch_views=args.input_ch_views, use_viewdirs=True)
teacher_model_fine.load_state_dict(saved['network_fine_state_dict'])
teacher_model_fine.eval()
print(teacher_model)

# NeRF class has been modified to track mins and maxes for all input
maxes = saved['maxes'].to(device)
print("maxes =", maxes)
mins = saved['mins'].to(device)
print("mins =", mins)

# Instantiate student models
student_model = NeRF(D=args.s_depth, W=args.s_width, input_ch=args.input_ch, input_ch_views=args.input_ch_views, skips=args.s_skips, use_viewdirs=True)
student_model_fine = NeRF(D=args.s_depth, W=args.s_width, input_ch=args.input_ch, input_ch_views=args.input_ch_views, skips=args.s_skips, use_viewdirs=True)
print(student_model)

num_params_teacher = 0
for param in teacher_model.parameters():
  num_params_teacher += param.numel()
num_params_student = 0
for param in student_model.parameters():
  num_params_student += param.numel()
print("Number of parameters in teacher network:", num_params_teacher, "\nNumber of parameters in student network:", num_params_student)
print("Size of student model: {:.2f}% of teacher model.".format((num_params_student/num_params_teacher)*100))

# Start of network distillation code
OUTPUT = 'O'
active_layers = [args.layer_queue.popleft()]
loss_over_time = []

# Send all models to device
student_model = student_model.to(device)
teacher_model = teacher_model.to(device)
student_model_fine = student_model_fine.to(device)
teacher_model_fine = teacher_model_fine.to(device)

# Use same optimizer for both student models
student_optim = torch.optim.Adam(list(student_model.parameters()) + list(student_model_fine.parameters()), lr=args.lr)

total_epochs = 0
while total_epochs < args.max_epochs and active_layers != []:
  for epoch in tqdm(range(args.status_freq)):
    # Generate random input
    rand_input = torch.rand(int(1024*64), args.input_ch + args.input_ch_views).to(device)
    rand_input = (maxes - mins) * rand_input + mins
    rand_input = rand_input.to(device)
    # Compute a forward pass
    # track_values=False tells the model not to track min and max values
    teacher_out = teacher_model(rand_input, track_values=False)
    student_out = student_model(rand_input, track_values=False)
    teacher_fine_out = teacher_model_fine(rand_input, track_values=False)
    student_fine_out = student_model_fine(rand_input, track_values=False)
    # Get hidden states from models
    teacher_out_hidden = teacher_model.hidden_states
    student_out_hidden = student_model.hidden_states
    teacher_fine_out_hidden = teacher_model_fine.hidden_states
    student_fine_out_hidden = student_model_fine.hidden_states

    # Compute loss as mse between active layers in both student and teacher models
    loss = torch.zeros(1).to(device)
    for layer_tuple in active_layers:
      if layer_tuple[0] == OUTPUT:
        student_layer = student_out
        student_fine_layer = student_fine_out
      else:
        student_layer = student_out_hidden[layer_tuple[0]]
        student_fine_layer = student_fine_out_hidden[layer_tuple[0]]

      if layer_tuple[1] == OUTPUT:
        teacher_layer = teacher_out
        teacher_fine_layer = teacher_fine_out
      else:
        teacher_layer = teacher_out_hidden[layer_tuple[1]]
        teacher_fine_layer = teacher_fine_out_hidden[layer_tuple[1]]

      loss += F.mse_loss(student_layer, teacher_layer) + F.mse_loss(student_fine_layer, teacher_fine_layer)

    # Backprop
    student_optim.zero_grad()
    loss.backward()
    student_optim.step()

    # Check to see if current active layers are within threshold
    if loss < args.loss_thresh:
      print("Completed layers: ", active_layers)
      fig, ax = plt.subplots(nrows=1, ncols=1)
      ax.plot(loss_over_time)
      ax.set_yscale('log')
      fig.savefig(args.plot_file.format(active_layers[0][0]))
      plt.close(fig)
      # Saving weights after each layer is finished
      model_save_path = "./logs/blender_paper_lego/student_model_{}.tar".format(active_layers[0][0]))
      if args.layer_queue:
        active_layers.append(args.layer_queue.popleft())
        #active_layers = [args.layer_queue.popleft()]
      else:
        active_layers = []
      torch.save({
          'global_step': saved['global_step'],
          'network_fn_state_dict': student_model.state_dict(),
          'network_fine_state_dict': student_model_fine.state_dict(),
          'optimizer_state_dict': student_optim.state_dict(),
      }, model_save_path)
      print("saved to ", model_save_path)

    # Plot loss according to log frequency
    if (total_epochs + epoch) % args.log_freq == 0:
      loss_over_time.append(loss)
  # end for epoch in tqdm

  total_epochs += epoch + 1
  # Print out a status according to frequency
  print("Epoch: {}, Loss: {}".format(total_epochs, loss.item()))
  print("Active layer:", active_layers)
  print("Layers in queue:", args.layer_queue)

# end while total_epochs < args.max_epochs and active_layers != []: