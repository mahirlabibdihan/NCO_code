from logging import getLogger

import torch
import random
from LEHD.CVRP.VRPModel import VRPModel as Model
from LEHD.CVRP.VRPEnv_inCVRPlib import VRPEnv as Env
from LEHD.utils.utils import *


# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class VRPTester():
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

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
        self.time_estimator.reset()
        self.time_estimator_2.reset()


        self.env.load_raw_data(self.tester_params['test_episodes'])

        score_AM = AverageMeter()
        score_student_AM = AverageMeter()

        test_num_episode = self.tester_params['test_episodes']
        episode = self.tester_params['begin_index']
        problems_100 = []
        problems_100_200 = []
        problems_200_500 = []
        problems_500_1000 = []
        problems_1000 = []

        problems_A = []
        problems_B = []
        problems_E = []
        problems_F = []
        problems_M = []
        problems_P = []
        problems_X = []

        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            score_teacher, score_student, problems_size, vrpname = self._test_one_batch(
                episode, batch_size, clock=self.time_estimator_2,logger = self.logger)
            current_gap = (score_student - score_teacher) / score_teacher
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


            if vrpname[:2]=='A-':
                problems_A.append(current_gap)
            elif vrpname[:2]=='B-':
                problems_B.append(current_gap)
            elif vrpname[:2]=='E-':
                problems_E.append(current_gap)
            elif vrpname[:2]=='F-':
                problems_F.append(current_gap)
            elif vrpname[:2]=='M-':
                problems_M.append(current_gap)
            elif vrpname[:2]=='P-':
                problems_P.append(current_gap)
            elif vrpname[:2]=='X-':
                problems_X.append(current_gap)

            print('problems_100 mean gap:', np.mean(problems_100), len(problems_100))
            print('problems_100_200 mean gap:', np.mean(problems_100_200), len(problems_100_200))
            print('problems_200_500 mean gap:', np.mean(problems_200_500), len(problems_200_500))
            print('problems_500_1000 mean gap:', np.mean(problems_500_1000), len(problems_500_1000))
            print('problems_1000 mean gap:', np.mean(problems_1000), len(problems_1000))

            self.logger.info(" problems_A    mean gap:{:4f}%, num:{}".format(np.mean( problems_A)*100,len( problems_A) ))
            self.logger.info(" problems_B    mean gap:{:4f}%, num:{}".format(np.mean( problems_B)*100,len( problems_B) ))
            self.logger.info(" problems_E    mean gap:{:4f}%, num:{}".format(np.mean( problems_E)*100, len(problems_E)))
            self.logger.info(" problems_F    mean gap:{:4f}%, num:{}".format(np.mean( problems_F)*100, len(problems_F)))
            self.logger.info(" problems_M    mean gap:{:4f}%, num:{}".format(np.mean( problems_M)*100, len(problems_M)))
            self.logger.info(" problems_P    mean gap:{:4f}%, num:{}".format(np.mean( problems_P)*100, len(problems_P)))
            self.logger.info(" problems_X    mean gap:{:4f}%, num:{}".format(np.mean( problems_X)*100, len(problems_X)))


            score_AM.update(score_teacher, batch_size)
            score_student_AM.update(score_student, batch_size)

            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], Score_teacher:{:.4f}, Score_studetnt: {:.4f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score_teacher, score_student))

            all_done = (episode == test_num_episode)

            if all_done:
                if self.env_params['test_in_vrplib']:
                    self.logger.info(" *** Test Done *** ")
                    all_result_gaps = problems_A + problems_B + problems_E + problems_F + problems_M + problems_P + problems_X
                    gap_ = np.mean(all_result_gaps)*100
                    self.logger.info(" Gap: {:.4f}%".format(gap_))
                else:
                    self.logger.info(" *** Test Done *** ")
                    self.logger.info(" Teacher SCORE: {:.4f} ".format(score_AM.avg))
                    self.logger.info(" Student SCORE: {:.4f} ".format(score_student_AM.avg))
                    gap_ = (score_student_AM.avg - score_AM.avg) / score_AM.avg * 100
                    self.logger.info(" Gap: {:.4f}%".format(gap_))

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
        acceptance_probability = torch.exp(-delta_reward / temperature).clamp(max=1.0).item()

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

    def get_routes(self, solution):
        problem_size = solution.shape[0]
        # Step 1: Identify routes based on the `flag` values (1 indicates start of a new route)
        routes = []  # This will store list of routes (each route is a list of indices)

        current_route = []
        for i in range(problem_size):
            if solution[i, 1] == 1:  # Customer is the start of a new route
                if current_route:  # If there are customers already in the current route, save it
                    routes.append(current_route)
                current_route = [i]  # Start a new route with the current customer
            else:
                current_route.append(i)  # Add customer to the current route
        if current_route:  # Add the last route if it ends at the last customer
            routes.append(current_route)
            
        return routes
                    
    def generate_neighbor(self, solution):
        batch_size = solution.shape[0]
        problem_size = solution.shape[1]
        
        # Create a clone of the solution to modify it
        neighbor_solution = solution.clone()

        for b in range(batch_size):
            # Extract the current solution for the batch
            current_solution = neighbor_solution[b]

            # Step 1: Identify routes based on the `flag` values (1 indicates start of a new route)
            routes = self.get_routes(current_solution)

            # Step 2: Pick a random route
            if routes:
                while True:
                    selected_route = random.choice(routes)  # Randomly choose a route
                    
                    # Ensure there are at least 2 customers in the route to swap
                    if len(selected_route) > 1:
                        # Step 3: Pick two random customers from the selected route
                        i, j = random.sample(range(len(selected_route)), 2)  # Pick two random indices from the route

                        # Step 4: Get the actual indices of the customers in the full solution
                        customer_i = selected_route[i]
                        customer_j = selected_route[j]

                        # Swap the two customers
                        neighbor_solution[b, customer_i, 0], neighbor_solution[b, customer_j, 0] = \
                            neighbor_solution[b, customer_j, 0], neighbor_solution[b, customer_i, 0]  # Swap the node values
                        break

        return neighbor_solution

    def is_valid_solution(self, solution):
        # Validate the solution by checking demand and capacity constraints
        # Iterate through each route (each sequence of nodes between 1 flags)
        batch_size = solution.shape[0]
        neighbor_solution = solution.clone()

        for b in range(batch_size):
            current_solution = neighbor_solution[b]
            # print(current_solution)
            for route in self.get_routes(current_solution):
                route_demand = 0
                for node in route:
                    route_demand += self.origin_problem[0, current_solution[node, 0], 2].item()  # Demand is at index 1 in the problem
                    
                route_capacity = self.origin_problem[0, 0, 3].item()  # Capacity of the vehicle is stored at index 2 of the depot node

                # If the demand exceeds the vehicle's capacity, the solution is invalid
                if route_demand > route_capacity:
                    for customer in route:
                        print(current_solution[customer, 0].item(), end=' ')
                    print(route_demand, route_capacity)
                    return False
        
        print(route_demand, route_capacity)
        return True

    def iterative_solution_improvement_sa(self, episode, clock, name, batch_size, current_step, best_select_node_list):
        budget = self.env_params['RRC_budget']

        # Simulated Annealing Parameters
        T_init = 100  # Initial temperature
        T_min = 1e-3  # Minimum temperature
        alpha = 0.95  # Cooling rate
        temperature = T_init

        # Track the best solution and its length
        best_solution = best_select_node_list.clone()  # Initial best solution
        best_solution_length = self.env._get_travel_distance_2(self.origin_problem, best_solution).mean().item()
        
        for bbbb in range(budget):
            # Clear CUDA cache to manage memory
            torch.cuda.empty_cache()

            # Reload problems
            self.env.load_problems(episode, batch_size)

            # Randomly sample and modify the partial solution
            # best_select_node_list = self.env.vrp_whole_and_solution_subrandom_inverse(best_select_node_list)
            print("Shape of best_select_node_list:", best_select_node_list.shape)
            
            if not self.is_valid_solution(best_select_node_list):
                raise ValueError("Invalid solution generated!")
            
            new_best_select_node_list  = self.generate_neighbor(best_select_node_list)

            if not self.is_valid_solution(new_best_select_node_list):
                raise ValueError("Invalid solution generated!")
                
            
            new_length = self.env._get_travel_distance_2(self.origin_problem, new_best_select_node_list)
            current_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list)
            delta_length = new_length.mean().item() - current_length.mean().item()
            
            # Simulated Annealing Acceptance Criteria
            if delta_length < 0 or torch.rand(1).item() < torch.exp(torch.tensor(-delta_length / temperature)):
            # Accept the new solution
                best_select_node_list = new_best_select_node_list
                
                # If this is the best solution found so far, update best_solution
                if new_length.mean().item() < best_solution_length:
                    best_solution = new_best_select_node_list
                    best_solution_length = new_length.mean().item()

            # Cool down the temperature
            temperature = max(T_min, temperature * alpha)

            # Get elapsed time
            escape_time, _ = clock.get_est_string(1, 1)

            # Log solution improvement details
            self.logger.info(
                "RRC step{}, name:{}, gap:{:6f} %, Elapsed[{}], stu_l:{:5f} , opt_l:{:5f}, Temp:{:5f}".format(
                    bbbb, name, ((current_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100,
                    escape_time, current_length.mean().item(), self.optimal_length.mean().item(), temperature))

        # Return the best solution found
        current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_solution)
        
        return current_best_length

    def _test_one_batch(self, episode, batch_size, clock=None,logger = None):

        random_seed = 12
        torch.manual_seed(random_seed)

        ###############################################
        self.model.eval()

        with torch.no_grad():

            self.env.load_problems(episode, batch_size)

            reset_state, _, _ = self.env.reset(self.env_params['mode'])

            current_step = 0

            self.origin_problem = self.env.problems.clone().detach()

            if self.env.test_in_vrplib:
                self.optimal_length, name  = self.env._get_travel_distance_2(self.origin_problem, self.env.solution,
                                                                      need_optimal=True)
            else:
                self.optimal_length= self.env._get_travel_distance_2(self.origin_problem, self.env.solution)
                name = 'vrp'+str(self.env.solution.shape[1])
                
            best_select_node_list, current_best_length = self.construct_initial_solution(batch_size, current_step)

            print('Get first complete solution!')

            escape_time, _ = clock.get_est_string(1, 1)

            self.logger.info("Greedy, name:{}, gap:{:5f} %, Elapsed[{}], stu_l:{:5f} , opt_l:{:5f}".format(name,
                ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100, escape_time,
            current_best_length.mean().item(), self.optimal_length.mean().item()))


            ####################################################

            # Iterative solution improvement
            current_best_length = self.iterative_solution_improvement_sa(
                episode, clock, name,  batch_size, current_step, best_select_node_list
            )
            # current_best_length = self.iterative_solution_improvement_sa(
            #     episode, clock, name,  batch_size, current_step, best_select_node_list
            # )
            
            print(f'current_best_length', (current_best_length.mean() - self.optimal_length.mean())
                  / self.optimal_length.mean() * 100, '%', 'escape time:', escape_time,
                  f'optimal:{self.optimal_length.mean()}, current_best:{current_best_length.mean()}')

            # 4. Cycle until the budget is consumed.
            # self.env.valida_solution_legal(self.origin_problem, best_select_node_list)

            # self.env.drawPic_VRP(self.origin_problem[0,:,[0,1]], best_select_node_list[0,:,0],best_select_node_list[0,:,1],name=name)

            return self.optimal_length.mean().item(), current_best_length.mean().item(), self.env.problem_size, name
