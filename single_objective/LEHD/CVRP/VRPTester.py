
from logging import getLogger
import torch
from LEHD.CVRP.VRPModel import VRPModel as Model
from LEHD.CVRP.VRPEnv import VRPEnv as Env
from LEHD.utils.utils import *
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class VRPTester():
    """
    A class to test Vehicle Routing Problem (VRP) models.
    Methods:
        run():
            Runs the testing process for the VRP model.
        decide_whether_to_repair_solution(after_repair_sub_solution, before_reward, after_reward, first_node_index, length_of_subpath, double_solution):
            Decides whether to repair the solution based on rewards and updates the solution accordingly.
        _test_one_batch(episode, batch_size, clock=None, logger=None):
            Tests one batch of episodes and logs the results.
    """
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):
        
        """
        Initialize the VRPTester class.

        Args:
            env_params (dict): Parameters for the environment.
            model_params (dict): Parameters for the model.
            tester_params (dict): Parameters for the tester, including:
                - use_cuda (bool): Whether to use CUDA for computation.
                - cuda_device_num (int): The CUDA device number to use.
                - model_load (dict): Information for loading the model, including:
                    - path (str): Path to the model checkpoint.
                    - epoch (int): Epoch number of the checkpoint.

        Attributes:
            env_params (dict): Parameters for the environment.
            model_params (dict): Parameters for the model.
            tester_params (dict): Parameters for the tester.
            logger (Logger): Logger instance for logging information.
            result_folder (str): Path to the folder for storing results.
            device (torch.device): Device to be used for computation (CPU or CUDA).
            env (Env): Environment instance created with env_params.
            model (Model): Model instance created with model_params.
            time_estimator (TimeEstimator): Utility for estimating time.
            time_estimator_2 (TimeEstimator): Another utility for estimating time.
        """

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()

        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        # Restore
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # utility
        self.time_estimator = TimeEstimator()
        self.time_estimator_2 = TimeEstimator()

    def run(self):
        # Reset time estimation utilities
        self.time_estimator.reset()
        self.time_estimator_2.reset()

        # Load raw test data based on specified number of test episodes
        self.env.load_raw_data(self.tester_params['test_episodes'])

        # Initialize average meters to track teacher and student scores
        score_AM = AverageMeter()
        score_student_AM = AverageMeter()

        # Get total number of test episodes
        test_num_episode = self.tester_params['test_episodes']

        # Initialize episode counter
        episode = 0

        # Initialize lists to track performance across different problem sizes
        problems_100 = []
        problems_100_200 = []
        problems_200_500 = []
        problems_500_1000 = []
        problems_1000 = []

        # Main testing loop
        while episode < test_num_episode:
            # Calculate remaining episodes and determine batch size
            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            # Run test for current batch
            score_teacher, score_student, problems_size = self._test_one_batch(
                episode, batch_size, clock=self.time_estimator_2, logger=self.logger)

            # Calculate performance gap for current batch
            current_gap = (score_student - score_teacher) / score_teacher

            # Categorize and store performance gap by problem size
            if problems_size < 100:
                problems_100.append(current_gap)
            elif 100 <= problems_size < 200:
                problems_100_200.append(current_gap)
            elif 200 <= problems_size < 500:
                problems_200_500.append(current_gap)
            elif 500 <= problems_size < 1000:
                problems_500_1000.append(current_gap)
            elif 1000 <= problems_size:
                problems_1000.append(current_gap)

            # Print mean performance gaps for each problem size category
            print('problems_100 mean gap:', np.mean(problems_100), len(problems_100))
            print('problems_100_200 mean gap:', np.mean(problems_100_200), len(problems_100_200))
            print('problems_200_500 mean gap:', np.mean(problems_200_500), len(problems_200_500))
            print('problems_500_1000 mean gap:', np.mean(problems_500_1000), len(problems_500_1000))
            print('problems_1000 mean gap:', np.mean(problems_1000), len(problems_1000))

            # Update average meters with current batch scores
            score_AM.update(score_teacher, batch_size)
            score_student_AM.update(score_student, batch_size)

            # Increment episode counter
            episode += batch_size

            # Log time and performance for current batch
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], Score_teacher:{:.4f}, Score_studetnt: {:.4f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score_teacher, score_student))

            # Check if all episodes have been processed
            all_done = (episode == test_num_episode)

            # If all episodes are done, log final results
            if all_done:
                self.logger.info(" *** Test Done *** ")
                self.logger.info(" Teacher SCORE: {:.4f} ".format(score_AM.avg))
                self.logger.info(" Student SCORE: {:.4f} ".format(score_student_AM.avg))
                
                # Calculate overall performance gap
                gap_ = (score_student_AM.avg - score_AM.avg) / score_AM.avg * 100
                self.logger.info(" Gap: {:.4f}%".format(gap_))

        # Return final scores and performance gap
        return score_AM.avg, score_student_AM.avg, gap_

    def decide_whether_to_repair_solution(self,
                                          after_repair_sub_solution, before_reward, after_reward,
                                          first_node_index, length_of_subpath, double_solution):

        the_whole_problem_size = int(double_solution.shape[1] / 2)
        batch_size = len(double_solution)

        temp = torch.arange(double_solution.shape[1])

        x3 = temp >= first_node_index[:, None].long()
        x4 = temp < (first_node_index[:, None] + length_of_subpath).long()
        x5 = x3 * x4

        origin_sub_solution = double_solution[x5.unsqueeze(2).repeat(1, 1, 2)].reshape(batch_size, length_of_subpath, 2)

        jjj, _ = torch.sort(origin_sub_solution[:, :, 0], dim=1, descending=False)

        index = torch.arange(batch_size)[:, None].repeat(1, jjj.shape[1])

        kkk_2 = jjj[index, after_repair_sub_solution[:, :, 0] - 1]

        after_repair_sub_solution[:, :, 0] = kkk_2

        if_repair = before_reward > after_reward

        need_to_repari_double_solution = double_solution[if_repair]
        need_to_repari_double_solution[x5[if_repair].unsqueeze(2).repeat(1, 1, 2)] = after_repair_sub_solution[if_repair].ravel()
        double_solution[if_repair] = need_to_repari_double_solution

        x6 = temp >= (first_node_index[:, None] + length_of_subpath - the_whole_problem_size).long()

        x7 = temp < (first_node_index[:, None] + length_of_subpath).long()

        x8 = x6 * x7

        after_repair_complete_solution = double_solution[x8.unsqueeze(2).repeat(1, 1, 2)].reshape(batch_size, the_whole_problem_size, -1)

        return after_repair_complete_solution

    def decide_whether_to_repair_solution_sa(self, after_repair_sub_solution, before_reward, after_reward, 
                                        first_node_index, length_of_subpath, double_solution, temperature):
        """
        Decide whether to replace the current solution with the repaired solution, 
        based on simulated annealing criteria.
        """

        # Calculate the whole problem size
        the_whole_problem_size = int(double_solution.shape[1] / 2)
        batch_size = len(double_solution)

        # Create a range tensor
        temp = torch.arange(double_solution.shape[1])

        # Create masks for the subpath
        x3 = temp >= first_node_index[:, None].long()
        x4 = temp < (first_node_index[:, None] + length_of_subpath).long()
        x5 = x3 * x4

        # Extract the original sub-solution
        origin_sub_solution = double_solution[x5.unsqueeze(2).repeat(1, 1, 2)].reshape(batch_size, length_of_subpath, 2)

        # Sort the original sub-solution
        jjj, _ = torch.sort(origin_sub_solution[:, :, 0], dim=1, descending=False)

        # Create an index tensor
        index = torch.arange(batch_size)[:, None].repeat(1, jjj.shape[1])

        # Map the sorted indices to the after repair sub-solution
        kkk_2 = jjj[index, after_repair_sub_solution[:, :, 0] - 1]

        # Update the after repair sub-solution
        after_repair_sub_solution[:, :, 0] = kkk_2

        # Calculate reward difference
        delta_reward = after_reward - before_reward

        # Simulated annealing acceptance probability
        acceptance_probability = torch.exp(-delta_reward / temperature).clamp(max=1.0)

        # Determine if repair is needed (always accept better solutions or probabilistically accept worse ones)
        if_repair = (before_reward > after_reward) | (torch.rand(1).item() < acceptance_probability)

        # Update the double solution if repair is needed
        need_to_repari_double_solution = double_solution[if_repair]
        need_to_repari_double_solution[x5[if_repair].unsqueeze(2).repeat(1, 1, 2)] = after_repair_sub_solution[if_repair].ravel()
        double_solution[if_repair] = need_to_repari_double_solution

        # Create masks for the complete solution
        x6 = temp >= (first_node_index[:, None] + length_of_subpath - the_whole_problem_size).long()
        x7 = temp < (first_node_index[:, None] + length_of_subpath).long()
        x8 = x6 * x7

        # Extract the complete solution after repair
        after_repair_complete_solution = double_solution[x8.unsqueeze(2).repeat(1, 1, 2)].reshape(batch_size, the_whole_problem_size, -1)

        # Return the complete solution after repair
        return after_repair_complete_solution
    
    def construct_initial_solution(self, batch_size, current_step):
        # Prepare initial state and get first step information
        state, reward, reward_student, done = self.env.pre_step()
        # Prepare batch volume
        B_V = batch_size * 1
        while not done:
            loss_node, selected_teacher, selected_student, selected_flag_teacher, selected_flag_student = \
                    self.model(state, self.env.selected_node_list, self.env.solution, current_step, raw_data_capacity=self.env.raw_data_capacity)  # 更新被选择的点和概率

            if current_step == 0:
                selected_flag_teacher = torch.ones(B_V, dtype=torch.int)
                selected_flag_student = selected_flag_teacher
            current_step += 1

            state, reward, reward_student, done = \
                self.env.step(selected_teacher, selected_student, selected_flag_teacher, selected_flag_student)


        best_select_node_list = torch.cat((self.env.selected_student_list.reshape(batch_size, -1, 1),
                                            self.env.selected_student_flag.reshape(batch_size, -1, 1)), dim=2)

        current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list)
            
        return best_select_node_list, current_best_length

    def construct_initial_solution_beam(self, batch_size, current_step, k = 5):
        # Prepare initial state and get first step information
        state, reward, reward_student, done = self.env.pre_step()
        # Prepare batch volume
        B_V = batch_size * 1
        
        all_selected_teacher = []
        all_selected_student = []
        all_selected_flag_teacher = []
        all_selected_flag_student = []

        # Initialize with 1 state
        states_k = [state]  # Start with one state
        beam_selected_teacher = []
        beam_selected_student = []
        beam_selected_flag_teacher = []
        beam_selected_flag_student = []
        
        beams = []  # To hold the new expanded states
        # Main solving loop
        while not done:
            # Initialize containers for the current step's output for each beam
            beam_selected_teacher.clear()
            beam_selected_student.clear()
            beam_selected_flag_teacher.clear()
            beam_selected_flag_student.clear()

            # For the first step, set all flags to 1 for each of the k solutions
            if current_step == 0:
                selected_flag_teacher = torch.ones(batch_size, k, dtype=torch.int)  # Shape: (B_V, k)
                selected_flag_student = selected_flag_teacher
                    
            for beam_idx in range(len(states_k)):  # Loop through the current states
                # Run the model for each beam and get selection probabilities
                loss_node, selected_teacher, selected_student, selected_flag_teacher, selected_flag_student, topk_probs = \
                    self.model(states_k[beam_idx], self.env.selected_node_list, self.env.solution, current_step,
                            raw_data_capacity=self.env.raw_data_capacity)

                # Store the results for the current beam
                beam_selected_teacher.append(selected_teacher)
                beam_selected_student.append(selected_student)
                beam_selected_flag_teacher.append(selected_flag_teacher)
                beam_selected_flag_student.append(selected_flag_student)

            # Increment step counter
            current_step += 1
            
            # Stack the results for the current step
            beam_selected_teacher = torch.stack(beam_selected_teacher, dim=1)  # Shape: (batch_size, k, solution_width)
            beam_selected_student = torch.stack(beam_selected_student, dim=1)  # Shape: (batch_size, k, solution_width)
            beam_selected_flag_teacher = torch.stack(beam_selected_flag_teacher, dim=1)  # Shape: (batch_size, k, solution_width)
            beam_selected_flag_student = torch.stack(beam_selected_flag_student, dim=1)  # Shape: (batch_size, k, solution_width)
            
            # Loop over all states (beams)
            for beam_idx in range(len(states_k)):
                selected_teacher_i = beam_selected_teacher[:, beam_idx]
                selected_student_i = beam_selected_student[:, beam_idx]
                selected_flag_teacher_i = beam_selected_flag_teacher[:, beam_idx]
                selected_flag_student_i = beam_selected_flag_student[:, beam_idx]

                # Call the environment step function for each solution path (beam_idx)
                state, reward, reward_student, done = self.env.step(
                    selected_teacher_i, selected_student_i, selected_flag_teacher_i, selected_flag_student_i
                )

                states_k.append(state)  # Expand with the new state

                # Store the selected nodes for the current solution
                all_selected_teacher.append(selected_teacher_i)
                all_selected_student.append(selected_student_i)
                all_selected_flag_teacher.append(selected_flag_teacher_i)
                all_selected_flag_student.append(selected_flag_student_i)

            
                    
        # After the loop, we have selected results for the solutions at each step
        all_selected_teacher = torch.stack(all_selected_teacher, dim=1)  # Shape: (batch_size, k, solution_width)
        all_selected_student = torch.stack(all_selected_student, dim=1)  # Shape: (batch_size, k, solution_width)
        all_selected_flag_teacher = torch.stack(all_selected_flag_teacher, dim=1)  # Shape: (batch_size, k, solution_width)
        all_selected_flag_student = torch.stack(all_selected_flag_student, dim=1)  # Shape: (batch_size, k, solution_width)

        # Initialize variables to track the best solution and its travel distance
        best_solution_idx = None
        best_travel_distance = float('inf')  # Start with a very large number

        # Loop through each solution (among k solutions)
        for i in range(len(states_k)):  # Loop over k states
            best_select_node_list = all_selected_student[:, i, :]  # Shape: [batch_size, solution_width]

            # Calculate the travel distance for the current solution
            current_travel_distance = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list)

            # If the current solution is better (has a shorter distance), update the best solution
            if current_travel_distance < best_travel_distance:
                best_travel_distance = current_travel_distance
                best_solution_idx = i  # Track the index of the best solution

        # Now, we have the best solution among the k solutions
        best_select_node_list = all_selected_student[:, best_solution_idx, :]

        # Calculate the length of the best solution
        current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list)

        return best_select_node_list, current_best_length
    
    def iterative_solution_improvement(self, episode, clock, name, batch_size, current_step, best_select_node_list):
        budget = self.env_params['RRC_budget']

        for bbbb in range(budget):
            torch.cuda.empty_cache()

            # 1. The complete solution is obtained, which corresponds to the problems of the current env

            self.env.load_problems(episode, batch_size)

            # 2. Sample the partial solution, reset env, and assign the first node and last node in env

            best_select_node_list = self.env.vrp_whole_and_solution_subrandom_inverse(best_select_node_list)

            partial_solution_length, first_node_index, length_of_subpath, double_solution = \
                self.env.destroy_solution(self.env.problems, best_select_node_list)

            before_repair_sub_solution = self.env.solution

            before_reward = partial_solution_length

            current_step = 0

            reset_state, _, _ = self.env.reset(self.env_params['mode'])

            state, reward, reward_student, done = self.env.pre_step()  # state: data, first_node = current_node

            # 3. Generate solution 2 again, compare the path lengths of solution 1 and solution 2,
            # and decide which path to accept.

            while not done:
                if current_step == 0:
                    selected_teacher = self.env.solution[:, 0, 0]
                    selected_flag_teacher = self.env.solution[:, 0, 1]
                    selected_student = selected_teacher
                    selected_flag_student = selected_flag_teacher


                else:
                    _, selected_teacher, selected_student, selected_flag_teacher, selected_flag_student = \
                        self.model(state, self.env.selected_node_list, self.env.solution, current_step,
                                    raw_data_capacity=self.env.raw_data_capacity)

                current_step += 1

                state, reward, reward_student, done = \
                    self.env.step(selected_teacher, selected_student, selected_flag_teacher, selected_flag_student)

            after_repair_sub_solution = torch.cat((self.env.selected_student_list.unsqueeze(2),
                                                    self.env.selected_student_flag.unsqueeze(2)), dim=2)

            after_reward = - reward_student

            after_repair_complete_solution = self.decide_whether_to_repair_solution(
                    after_repair_sub_solution,
                before_reward, after_reward, first_node_index, length_of_subpath, double_solution)

            best_select_node_list = after_repair_complete_solution

            current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list)

            escape_time, _ = clock.get_est_string(1, 1)

            self.logger.info(
                "RRC step{}, name:{}, gap:{:6f} %, Elapsed[{}], stu_l:{:5f} , opt_l:{:5f}".format(
                        bbbb, name, ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100,
                    escape_time,current_best_length.mean().item(), self.optimal_length.mean().item()))

        current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list)
        
        return current_best_length


    def iterative_solution_improvement_sa_rrc(self, episode, clock, name, batch_size, current_step, best_select_node_list):
        budget = self.env_params['RRC_budget']

        # Simulated Annealing Parameters
        # Iteration: 100
        T_init = 100  # Initial temperature
        T_min = 1e-3  # Minimum temperature
        alpha = 0.98  # Cooling rate
        temperature = T_init
        
        best_solution = best_select_node_list.clone()  # Initial best solution
        best_solution_length = self.env._get_travel_distance_2(self.origin_problem, best_solution).mean()

        for bbbb in range(budget):
            # Clear CUDA cache to manage memory
            torch.cuda.empty_cache()

            # Reload problems
            self.env.load_problems(episode, batch_size)

            # Randomly sample and modify the partial solution
            best_select_node_list = self.env.vrp_whole_and_solution_subrandom_inverse(best_select_node_list)

            # Destroy and partially reconstruct the solution
            partial_solution_length, first_node_index, length_of_subpath, double_solution = \
                self.env.destroy_solution(self.env.problems, best_select_node_list)

            # Store solution before repair
            before_repair_sub_solution = self.env.solution
            before_reward = partial_solution_length

            # Reset environment and prepare for solution reconstruction
            current_step = 0
            reset_state, _, _ = self.env.reset(self.env_params['mode'])
            state, reward, reward_student, done = self.env.pre_step()

            # Solution reconstruction loop
            while not done:
                # For the first step, use initial solution nodes
                if current_step == 0:
                    selected_teacher = self.env.solution[:, 0, 0]
                    selected_flag_teacher = self.env.solution[:, 0, 1]
                    selected_student = selected_teacher
                    selected_flag_student = selected_flag_teacher
                else:
                    # Run model to select nodes for reconstruction
                    _, selected_teacher, selected_student, selected_flag_teacher, selected_flag_student = \
                        self.model(state, self.env.selected_node_list, self.env.solution, current_step,
                                raw_data_capacity=self.env.raw_data_capacity)

                current_step += 1

                # Take a step in the environment
                state, reward, reward_student, done = \
                    self.env.step(selected_teacher, selected_student, selected_flag_teacher, selected_flag_student)

            # Prepare reconstructed solution
            after_repair_sub_solution = torch.cat((self.env.selected_student_list.unsqueeze(2),
                                                self.env.selected_student_flag.unsqueeze(2)), dim=2)

            after_reward = - reward_student

            # Decide whether to keep the repaired solution using Simulated Annealing
            after_repair_complete_solution = self.decide_whether_to_repair_solution_sa(
                after_repair_sub_solution,
                before_reward, after_reward, first_node_index, length_of_subpath, double_solution, temperature
            )
            
            print(after_repair_complete_solution)
            best_select_node_list = after_repair_complete_solution

            current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list)
            
            
            # If this is the best solution found so far, update best_solution
            if current_best_length.mean().item() < best_solution_length.item():
                best_solution = best_select_node_list
                best_solution_length = current_best_length.mean()

            # Cool down the temperature
            temperature = max(T_min, temperature * alpha)

            # Get elapsed time
            escape_time, _ = clock.get_est_string(1, 1)

            # Log solution improvement details
            self.logger.info(
                "RRC step{}, name:{}, gap:{:6f} %, Elapsed[{}], stu_l:{:5f} , opt_l:{:5f}, Temp:{:5f}".format(
                    bbbb, name, ((best_solution_length - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100,
                    escape_time, best_solution_length.item(), self.optimal_length.mean().item(), temperature))

        # Final solution length calculation
        # current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list)
        
        return best_solution_length
    
    def _test_one_batch(self, episode, batch_size, clock=None, logger=None):
        """
        Test a batch of Vehicle Routing Problems (VRP) using a machine learning model.
        
        Args:
            episode (int): Identifier for the current test batch
            batch_size (int): Number of problems to solve simultaneously
            clock (object, optional): Timing utility for tracking elapsed time
            logger (object, optional): Logging utility for recording results
        
        Returns:
            tuple: (optimal solution length, current best solution length, problem size)
        """
        # Set a fixed random seed for reproducibility
        random_seed = 12
        torch.manual_seed(random_seed)

        # Set the model to evaluation mode
        self.model.eval()

        # Disable gradient calculation to reduce memory usage and computation
        with torch.no_grad():
            # Load problems for the current episode
            self.env.load_problems(episode, batch_size)

            # Reset the environment to its initial state
            reset_state, _, _ = self.env.reset(self.env_params['mode'])

            # Initialize step counter
            current_step = 0

            # Store the original problem for comparison
            self.origin_problem = self.env.problems.clone().detach()

            # Calculate the optimal travel distance for the original problem
            self.optimal_length = self.env._get_travel_distance_2(self.origin_problem, self.env.solution)

            # Create a problem name based on solution shape
            name = 'vrp'+str(self.env.solution.shape[1])

            best_select_node_list, current_best_length = self.construct_initial_solution(batch_size, current_step)
            print('Get first complete solution!')
            
            # Get elapsed time
            escape_time, _ = clock.get_est_string(1, 1)

            # Log initial solution details
            self.logger.info("Greedy, name:{}, gap:{:5f} %, Elapsed[{}], stu_l:{:5f} , opt_l:{:5f}".format(name,
                ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100, escape_time,
                current_best_length.mean().item(), self.optimal_length.mean().item()))

            # Iterative solution improvement
            current_best_length = self.iterative_solution_improvement_sa_rrc(
                episode, clock, name,  batch_size, current_step, best_select_node_list
            )
            
            # Print final results
            print(f'current_best_length', (current_best_length.mean() - self.optimal_length.mean())
                / self.optimal_length.mean() * 100, '%', 'escape time:', escape_time,
                f'optimal:{self.optimal_length.mean()}, current_best:{current_best_length.mean()}')

            # Return optimal length, current best length, and problem size
            return self.optimal_length.mean().item(), current_best_length.mean().item(), self.env.problem_size