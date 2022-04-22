import os
import sys
import argparse
import resource
import psutil

local_rank = int(os.environ.get("LOCAL_RANK", "0"))
os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % local_rank
import torch
import torchvision
import torchvision.transforms as transforms
import torch.distributed as tdist
from torchvision import models
from torchvision import transforms as transforms
import time
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

prof_running=False
def dumpru(ru,lastrss):
  global prof_running
  rss=psutil.Process(os.getpid()).memory_info().rss
  r=f"utime={ru.ru_utime:.3f} stime={ru.ru_stime:.3f} maxrss={ru.ru_maxrss/1024:.3f} MBs curr_rss={rss/(1024*1024):.3f} MBs DeltaRSS={(ru.ru_maxrss-lastrss[-1][1])/1024:.3f} MBs"
  lastrss.append((prof_running,ru.ru_maxrss))
  return r

def setup(rank, world_size):
  # os.environ['MASTER_ADDR'] = 'localhost'
  # os.environ['MASTER_PORT'] = '45654'
  tdist.init_process_group('nccl', rank=rank, world_size=world_size)


def cleanup():
  tdist.destroy_process_group()


# def find_world_size():
#   if not tdist.is_available():
#     return 1
#   if tdist.is_initialized():
#     return

def main():
  global prof_running
  transform = transforms.Compose([
      transforms.Resize(256),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ])

  #print("Running on ", torch.cuda.get_device_name(), "local_rank", local_rank)
  #print(os.environ)
  world_size = int(os.environ.get("WORLD_SIZE", "1"))
  parser = argparse.ArgumentParser("test")
  parser.add_argument("--profile", '-p', action='store_true')
  parser.add_argument("--num_steps", "-n", default=300, type=int)
  parser.add_argument("--batch_size", "-b", default=1, type=int)
  parser.add_argument("--epochs", "-e", default=6, type=int)
  parser.add_argument("--profiling_stop", "-s", default=3, type=int)
  parser.add_argument("--profiling_start", "-S", default=1, type=int)
  args, unk = parser.parse_known_args()

  sampler = None
  if world_size > 1:
    setup(local_rank, world_size)
  if tdist.is_initialized():
    if tdist.get_rank() == 0:
      trainset = torchvision.datasets.CIFAR10(root="./data",
                                              train=True,
                                              download=True,
                                              transform=transform)
    tdist.barrier()
    trainset = torchvision.datasets.CIFAR10(root="./data",
                                            train=True,
                                            download=False,
                                            transform=transform)
    sampler = torch.utils.data.distributed.DistributedSampler(trainset)
  else:
    trainset = torchvision.datasets.CIFAR10(root="./data",
                                            train=True,
                                            download=True,
                                            transform=transform)

  trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_size=args.batch_size,
                                            shuffle=(sampler is None),
                                            num_workers=8,
                                            drop_last=True,
                                            pin_memory=True,
                                            sampler=sampler,
                                            persistent_workers=True)

  net = models.resnet101(pretrained=False)
  net = net.to("cuda")
  if world_size > 1:
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[0])

  criterion = torch.nn.CrossEntropyLoss().cuda()
  optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

  print("ARGS=", args)

  doProfile = args.profile

  if doProfile:
      lastrss=[(prof_running,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)]
      print(f"Resource utilization before import rank[ {local_rank} ]=",
            dumpru(resource.getrusage(resource.RUSAGE_SELF),lastrss))

      import cupti_bug
      CCapture = cupti_bug.CuptiCapture.instance()
      for epoch in range(args.epochs):
        if epoch>=args.profiling_start and epoch<=args.profiling_stop:
          if not prof_running:
            CCapture.start_profiling()
            prof_running=True
        if world_size > 1:
          sampler.set_epoch(epoch)
        tstart = time.perf_counter()
        for i, data in enumerate(trainloader, 0):
          inputs, labels = data
          inputs = inputs.to("cuda", non_blocking=True)
          labels = labels.to("cuda", non_blocking=True)

          # zero the parameter gradients
          optimizer.zero_grad()

          outputs = net(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
          if i == args.num_steps:
            break
        if local_rank == 0:
          print("Time for", i, "steps of batch size", args.batch_size, "=",
                time.perf_counter() - tstart)
        print(f"End of epoch {epoch} rank[ {local_rank} ] capture[ {prof_running} ]  resource utilization =",
              dumpru(resource.getrusage(resource.RUSAGE_SELF),lastrss))
        if prof_running and epoch >= args.profiling_stop:
          print("Stopping cuda capture")
          prof_running=False
          CCapture.stop_profiling()

      if prof_running:
        CCapture.stop_profiling()

  else:
    for epoch in range(args.epochs):
      if world_size > 1:
        sampler.set_epoch(epoch)
      tstart = time.perf_counter()
      for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.to("cuda", non_blocking=True)
        labels = labels.to("cuda", non_blocking=True)

        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        # if (i % 10 == 0):
        #   print("GPU=", local_rank, "Step=", i, "loss=", loss.item())
        optimizer.step()

        if i == args.num_steps:
          break
      if local_rank == 0:
        print("Time for", i, "steps of batch size", args.batch_size, "=",
              time.perf_counter() - tstart)
  print(f"Resource utilization rank[ {local_rank} ]=",
        dumpru(resource.getrusage(resource.RUSAGE_SELF),lastrss))
  message=[]
  message.append(f"Memory usage before framework starts and cupti is imported = {lastrss[0][1]/(1024*1024):.3f} MB")
  for i,l in enumerate(lastrss[1:]):
    ac="RUNNING" if l[0] else "not running"
    message.append(f"Memory usage at the end of epoch {i} while cupti was {ac} ={l[1]/(1024*1024):.3f} MB. Change from previous = {(l[1]-lastrss[i][1])/(1024*1024):.3f} MB")
  print(f"Rank {local_rank} memory statistics: ","\n".join(message))
main()
