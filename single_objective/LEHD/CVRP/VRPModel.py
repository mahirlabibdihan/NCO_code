import torch
import torch.nn as nn
import torch.nn.functional as F


def top_k_sampling(logits, k):
    """
    Perform Top-K Sampling.
    Args:
        logits (torch.Tensor): Logits from the model, shape (batch_size, vocab_size).
        k (int): Number of top elements to consider.
    Returns:
        selected_indices (torch.Tensor): Selected token indices, shape (batch_size,).
    """
    # Get top-k logits and their indices
    top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)

    # Apply softmax to the top-k logits
    top_k_probs = F.softmax(top_k_logits, dim=-1)

    # Sample from the top-k probabilities
    selected_indices = torch.multinomial(top_k_probs, num_samples=1).squeeze(-1)

    # Map the selected indices back to the original vocabulary indices
    final_indices = top_k_indices.gather(-1, selected_indices.unsqueeze(-1)).squeeze(-1)
    return final_indices

def nucleus_sampling(logits, p=0.9):
    """
    Perform Nucleus Sampling (Top-p Sampling).
    
    Args:
        logits (torch.Tensor): Logits from the model, shape (batch_size, vocab_size).
        p (float): Probability threshold for nucleus sampling (usually 0.8 to 0.95).
        
    Returns:
        selected_indices (torch.Tensor): Selected token indices, shape (batch_size,).
    """
    # Convert logits to probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Sort probabilities in descending order and get their indices
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    
    # Compute the cumulative sum of the sorted probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Find the cutoff where the cumulative probability exceeds p
    cutoff_index = (cumulative_probs > p).float().sum(dim=-1).long()
    
    # Mask probabilities beyond the cutoff index (set them to 0)
    mask = torch.arange(sorted_probs.size(-1), device=logits.device)[None, :] < cutoff_index.unsqueeze(-1)
    sorted_probs = sorted_probs * mask.float()
    
    # Normalize the remaining probabilities
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    
    # Sample from the truncated distribution
    selected_indices = torch.multinomial(sorted_probs, 1).squeeze(-1)
    
    # Map back to the original indices
    selected_indices = torch.gather(sorted_indices, -1, selected_indices.unsqueeze(-1)).squeeze(-1)
    
    return selected_indices

class VRPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.mode = model_params['mode']
        self.encoder = CVRP_Encoder(**model_params)
        self.decoder = CVRP_Decoder(**model_params)

        self.encoded_nodes = None

    def greedy_decode(self, probs, split_line, batch_size):
        """
        Greedy decoding: Select the node with the highest probability.
        """
        selected_node_student = probs.argmax(dim=1)  # Shape: B -- Greedy Decoding

        # Determine if the selected nodes are via depot
        is_via_depot_student = selected_node_student >= split_line
        not_via_depot_student = selected_node_student < split_line

        # Set the flags for selected nodes
        selected_flag_student = torch.zeros(batch_size, dtype=torch.int)
        selected_flag_student[is_via_depot_student] = 1
        selected_node_student[is_via_depot_student] -= split_line - 1  # Adjust depot nodes' index
        selected_flag_student[not_via_depot_student] = 0
        selected_node_student[not_via_depot_student] += 1  # Adjust non-depot nodes' index

        # Return selected nodes and flags
        return selected_node_student, selected_flag_student
    
    def beam_decode(self, probs, split_line, batch_size, beam_width=5):
        """
        Beam search decoding: Select the top-k nodes with the highest probabilities.
        """
        topk_probs, topk_indices = torch.topk(probs, beam_width, dim=1, largest=True, sorted=False)
        
        # Create arrays to hold the selected nodes for each beam
        selected_node_student = topk_indices  # Shape: (batch_size, beam_width)
        
        # Determine if the selected nodes are via depot
        is_via_depot_student = selected_node_student >= split_line
        not_via_depot_student = selected_node_student < split_line
        
        # Initialize the flags for selected nodes for each beam
        selected_flag_student = torch.zeros(batch_size, beam_width, dtype=torch.int)
        
        # Update flags and selected nodes based on depot check
        selected_flag_student[is_via_depot_student] = 1
        selected_flag_student[not_via_depot_student] = 0
        selected_node_student[is_via_depot_student] -= split_line - 1  # Adjust indices for depot
        selected_node_student[not_via_depot_student] += 1  # Adjust indices for non-depot nodes
        
        return selected_node_student, selected_flag_student, topk_probs

    def forward_test(self, state, selected_node_list, current_step, split_line, batch_size, decoding_strategy, beam_width=5):
        """
        Main function to select decoding strategy: either 'greedy' or 'beam'.
        """
        remaining_capacity = state.problems[:, 1, 3]

        # Encode nodes if it's the first step
        if current_step <= 1:
            self.encoded_nodes = self.encoder(state.problems, self.capacity)

        # Get probabilities from the decoder
        probs = self.decoder(self.encoded_nodes, selected_node_list, self.capacity, remaining_capacity)

        # Choose the decoding strategy
        if decoding_strategy == 'greedy':
            selected_node_student, selected_flag_student = self.greedy_decode(probs, split_line, batch_size)
        elif decoding_strategy == 'beam':
            selected_node_student, selected_flag_student, topk_probs = self.beam_decode(probs, split_line, batch_size, beam_width)

        # Set the teacher's selected nodes and flags to the student's for each beam
        selected_node_teacher = selected_node_student.clone()
        selected_flag_teacher = selected_flag_student.clone()

        # Set the loss to zero for testing
        loss_node = torch.tensor(0)

        return loss_node, selected_node_teacher, selected_node_student, selected_flag_teacher, selected_flag_student, topk_probs
        
    def forward(self, state, selected_node_list, solution, current_step, decoding_strategy, raw_data_capacity=None):
        # solution's shape : [B, k, V]
        
        # Set the capacity from raw_data_capacity
        self.capacity = raw_data_capacity.ravel()[0].item()
        
        # Get the batch size and problem size
        batch_size = state.problems.shape[0]
        problem_size = state.problems.shape[1]
        
        # Define the split line
        split_line = problem_size - 1

        # Define a helper function to convert probabilities to selected nodes
        def probs_to_selected_nodes(probs_, split_line_, batch_size_):
            selected_node_student_ = probs_.argmax(dim=1)  # shape: B
            is_via_depot_student_ = selected_node_student_ >= split_line_  # Nodes with an index greater than customer_num are via depot
            not_via_depot_student_ = selected_node_student_ < split_line_

            selected_flag_student_ = torch.zeros(batch_size_, dtype=torch.int)
            selected_flag_student_[is_via_depot_student_] = 1
            selected_node_student_[is_via_depot_student_] = selected_node_student_[is_via_depot_student_] - split_line_ + 1
            selected_flag_student_[not_via_depot_student_] = 0
            selected_node_student_[not_via_depot_student_] = selected_node_student_[not_via_depot_student_] + 1
            return selected_node_student_, selected_flag_student_  # Node index starts from 1

        # Training mode
        if self.mode == 'train':
            remaining_capacity = state.problems[:, 1, 3]
            
            # Get probabilities from the decoder
            probs = self.decoder(self.encoder(state.problems, self.capacity),
                                 selected_node_list, self.capacity, remaining_capacity)
            
            # Convert probabilities to selected nodes and flags
            selected_node_student, selected_flag_student = probs_to_selected_nodes(probs, split_line, batch_size)
            
            # Get the teacher's selected nodes and flags
            selected_node_teacher = solution[:, current_step, 0]
            selected_flag_teacher = solution[:, current_step, 1]
            
            # Adjust the teacher's selected nodes for depot
            is_via_depot = selected_flag_teacher == 1
            selected_node_teacher_copy = selected_node_teacher - 1
            selected_node_teacher_copy[is_via_depot] += split_line
            
            # Calculate the loss for node selection
            prob_select_node = probs[torch.arange(batch_size)[:, None], selected_node_teacher_copy[:, None]].reshape(batch_size, 1)  # shape: [B, 1]
            loss_node = -prob_select_node.type(torch.float64).log().mean()

        # Testing mode
        if self.mode == 'test':
            print("Decoding strategy:", decoding_strategy)
            return self.forward_test(state, selected_node_list, current_step, split_line, batch_size, decoding_strategy)
        
        # Return the loss and selected nodes and flags for both teacher and student
        return loss_node, selected_node_teacher, selected_node_student, selected_flag_teacher, selected_flag_student




class CVRP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num =  1
        self.embedding = nn.Linear(3, embedding_dim, bias=True)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, data_,capacity):

        data = data_.clone().detach()
        data= data[:,:,:3]

        data[:,:,2] = data[:,:,2]/capacity

        embedded_input = self.embedding(data)

        out = embedded_input  # [B*(V-1), problem_size - current_step +2, embedding_dim]

        layer_count = 0
        for layer in self.layers:
            out = layer(out)
            layer_count += 1
        return out


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.feedForward = Feed_Forward_Module(**model_params)


    def forward(self, input1):

        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)

        out_concat = multi_head_attention(q, k, v)  # shape: (B, n, head_num*key_dim)

        multi_head_out = self.multi_head_combine(out_concat)  # shape: (B, n, embedding_dim)

        out1 = input1 + multi_head_out
        out2 = self.feedForward(out1)

        out3 = out1 + out2
        return out3
        # shape: (batch, problem, EMBEDDING_DIM)

########################################
# DECODER
########################################

class CVRP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        decoder_layer_num = self.model_params['decoder_layer_num']

        self.embedding_first_node = nn.Linear(embedding_dim+1, embedding_dim, bias=True)
        self.embedding_last_node = nn.Linear(embedding_dim+1, embedding_dim, bias=True)

        self.layers = nn.ModuleList([DecoderLayer(**model_params) for _ in range(decoder_layer_num)])
        self.Linear_final = nn.Linear(embedding_dim, 2, bias=True)

    def _get_new_data(self, data, selected_node_list, prob_size, B_V):
        list = selected_node_list
        new_list = torch.arange(prob_size)[None, :].repeat(B_V, 1)
        new_list_len = prob_size - list.shape[1]  # shape: [B, V-current_step]
        
        index_2 = list.type(torch.long)
        index_1 = torch.arange(B_V, dtype=torch.long)[:, None].expand(B_V, index_2.shape[1])
        
        # print("new_list shape before modification:", new_list.shape)
        # print("new_list before modification:", new_list)

        new_list[index_1, index_2] = -2
        
        # print("new_list after setting -2 for selected nodes:", new_list)
        
        unselect_list = new_list[torch.ne(new_list, -2)]
        
        # Print the shape and verify the number of elements
        # print(f"Expected unselect_list size: {B_V * new_list_len}")
        # print(f"Actual unselect_list size: {unselect_list.numel()}")
        
        # Handle the mismatch in unselect_list size
        if unselect_list.numel() != B_V * new_list_len:
            print(f"Warning: Shape mismatch! unselect_list has {unselect_list.numel()} elements, expected {B_V * new_list_len}.")
            
            # Handle mismatch by filling the remaining elements with some default value (e.g., zero)
            # You could also choose another default behavior depending on your needs.
            missing_elements = B_V * new_list_len - unselect_list.numel()
            if missing_elements > 0:
                unselect_list = torch.cat([unselect_list, torch.zeros(missing_elements, dtype=torch.long)])
            elif missing_elements < 0:
                unselect_list = unselect_list[:B_V * new_list_len]  # Trim extra elements
        
        unselect_list = unselect_list.view(B_V, new_list_len)
        
        new_data = data
        emb_dim = data.shape[-1]
        new_data_len = new_list_len
        
        # Ensure proper indexing for the new data
        index_2_ = unselect_list.repeat_interleave(repeats=emb_dim, dim=1)
        index_1_ = torch.arange(B_V, dtype=torch.long)[:, None].expand(B_V, index_2_.shape[1])
        index_3_ = torch.arange(emb_dim)[None, :].repeat(repeats=(B_V, new_data_len))
        
        new_data_ = new_data[index_1_, index_2_, index_3_].view(B_V, new_data_len, emb_dim)
        
        return new_data_


    def _get_encoding(self,encoded_nodes, node_index_to_pick):

        batch_size = node_index_to_pick.size(0)
        pomo_size = node_index_to_pick.size(1)
        embedding_dim = encoded_nodes.size(2)

        gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)

        picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)

        return picked_nodes


    def forward(self, data,selected_node_list,capacity,remaining_capacity):

        data_ = data[:,1:,:].clone().detach()
        selected_node_list_ = selected_node_list.clone().detach() - 1

        batch_size_V = data_.shape[0]  # B

        problem_size = data_.shape[1]

        new_data = data_.clone().detach()

        left_encoded_node = self._get_new_data(new_data, selected_node_list_, problem_size, batch_size_V)

        embedded_first_node = data[:,[0],:]

        if selected_node_list_.shape[1]==0:
            embedded_last_node = data[:,[0],:]
        else:
            embedded_last_node = self._get_encoding(new_data, selected_node_list_[:, [-1]])

        remaining_capacity = remaining_capacity.reshape(batch_size_V,1,1)/capacity
        first_node_cat = torch.cat((embedded_first_node,remaining_capacity), dim=2)
        last_node_cat = torch.cat((embedded_last_node,remaining_capacity), dim=2)
        # ------------------------------------------------
        # ------------------------------------------------

        embedded_first_node_ = self.embedding_first_node(first_node_cat)

        embedded_last_node_ = self.embedding_last_node(last_node_cat)


        embeded_all = torch.cat((embedded_first_node_,left_encoded_node,embedded_last_node_), dim=1)
        out = embeded_all  # [B*(V-1), problem_size - current_step +2, embedding_dim]

        layer_count = 0

        for layer in self.layers:

            out = layer(out)
            layer_count += 1


        out = self.Linear_final(out)  # shape: [B*(V-1), reminding_nodes_number + 2, embedding_dim ]

        out[:, [0, -1], :] = out[:, [0, -1], :] + float('-inf')  # first node、last node

        out = torch.cat((out[:, :, 0], out[:, :, 1]), dim=1)  # shape:(B, 2 * ( V - current_step ))

        props = F.softmax(out, dim=-1)
        customer_num = left_encoded_node.shape[1]

        props = torch.cat((props[:, 1:customer_num + 1], props[:, customer_num + 1 + 1 + 1:-1]),
                          dim=1)

        index_small = torch.le(props, 1e-5)
        props_clone = props.clone()
        props_clone[index_small] = props_clone[index_small] + torch.tensor(1e-7, dtype=props_clone[index_small].dtype)
        props = props_clone

        new_props = torch.zeros(batch_size_V, 2 * (problem_size))

        # The function of the following part is to fill the probability of props into the new_props,
        index_1_ = torch.arange(batch_size_V, dtype=torch.long)[:,None].repeat(1,selected_node_list_.shape[1]*2)
        index_2_ =torch.cat( ((selected_node_list_).type(torch.long), (problem_size)+ (selected_node_list_).type(torch.long) ),dim=-1) # shape: [B*V, n]
        new_props[index_1_, index_2_,] = -2
        index = torch.gt(new_props, -1).view(batch_size_V, -1)
        
        # Value Error
        padding_size = new_props[index].shape[0] - props.ravel().shape[0]
        padded_props = torch.cat([props.ravel(), torch.zeros(padding_size)])  # Pad with zeros
        new_props[index] = padded_props
        # new_props[index] = props.ravel()

        return new_props


class DecoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.feedForward = Feed_Forward_Module(**model_params)

    def forward(self, input1):
        head_num = self.model_params['head_num']
        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        out_concat = multi_head_attention(q, k, v)
        multi_head_out = self.multi_head_combine(out_concat)
        out1 = input1 + multi_head_out
        out2 = self.feedForward(out1)
        out3 = out1 + out2
        return out3

def reshape_by_heads(qkv, head_num):
    batch_s = qkv.size(0)
    n = qkv.size(1)
    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    q_transposed = q_reshaped.transpose(1, 2)
    return q_transposed

def multi_head_attention(q, k, v):
    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    score = torch.matmul(q, k.transpose(2, 3))  # shape: (B, head_num, n, n)
    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    weights = nn.Softmax(dim=3)(score_scaled)  # shape: (B, head_num, n, n)
    out = torch.matmul(weights, v)  # shape: (B, head_num, n, key_dim)
    out_transposed = out.transpose(1, 2)  # shape: (B, n, head_num, key_dim)
    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)  # shape: (B, n, head_num*key_dim)
    return out_concat

class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        return self.W2(F.relu(self.W1(input1)))
