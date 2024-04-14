import os
import torch
import torch.distributed as dist
import threading
import copy


HEAD_TYPES = [0, 1]

class NewKVCacheCommManager:

    def __init__(self, rank, world_size, num_prompt_workers, device) -> None:
        self.corr_worker_rank = (rank + num_prompt_workers) % world_size
        self.rank = rank
        self.remote_rank = self.corr_worker_rank
        self.device = device
        

    def setup_comm(self, num_layers, kv_cache, block_size) -> None:
        # Set up proxy service and proxy channels for KV cache communication.
        # TODO: key and value have different shapes. Fix the shape later for receiver
        self.kv_cache=kv_cache
        tmp_shape = kv_cache[0][0][0].shape
        self.recv_shape= tuple(dim + 1 if i == 0 else dim for i, dim in enumerate(tmp_shape))
        self.num_layers=num_layers
        self.block_size=block_size
        self.recv_data_length= self.block_size+1

    def warm_up(self, isTokenWorker)->None:
        print("warming up")
        print(self.device)

        if not isTokenWorker:
            print("sneding")
            tensor_to_send = torch.ones(2, device=self.device)
            req_send = dist.isend(tensor=tensor_to_send, dst=self.remote_rank)
            req_send.wait() 
            print("sent")
            
        else:
            print("waiting!!!")
            tensor_to_receive = torch.zeros(2, device=self.device)
            req_recv = dist.irecv(tensor=tensor_to_receive, src=self.remote_rank)
            req_recv.wait()  
            print("tensor_to_receive  ",tensor_to_receive)

        dist.barrier()
        torch.cuda.synchronize()
        print("done warm up", self.rank, self.device)


    def wait(self, sem_id):
        print(self.device)
        recv_data = torch.zeros(self.recv_shape, device=self.device)
        req_recv = dist.irecv(tensor=recv_data, src=self.remote_rank)
        req_recv.wait()
        lastItem = recv_data[-1].view(-1)
        layer_id=lastItem[0].item()
        head_type=lastItem[1].item()
        block_tensor_start=lastItem[2].item()
        self.kv_cache[layer_id][head_type][block_tensor_start]=recv_data[:-1]

    def addDetails(self, layer_id, head_type, block_tensor_start, tensorData):
        flattened_tensor = tensorData.view(-1)
        new_values = torch.tensor([layer_id, head_type, block_tensor_start], device=self.device)
        flattened_tensor[:3] = new_values
        return flattened_tensor.view(tensorData.shape)

    def send_message(self, semid, layer_id, block_start, num_blocks ):
        for head_type in HEAD_TYPES:
            for blockI in range(num_blocks):
                block_tensor_id = block_start+blockI
                # tried clone() but not working too
                data = copy.deepcopy(self.kv_cache[layer_id][head_type][block_tensor_id])
                sampleBlock=copy.deepcopy(data[0])
                modifiedData =self.addDetails(layer_id, head_type,block_tensor_id,sampleBlock) 
                modifiedData = modifiedData.unsqueeze(0)
                data = torch.cat((data,modifiedData), dim=0)
                dist.isend(tensor=data, dst=self.remote_rank)
    
    def put(self, semid, layer_id, block_start, num_blocks ):
        send_thread = threading.Thread(target=self.send_message, args=(semid, layer_id, block_start, num_blocks ))
        send_thread.start()
        


    


