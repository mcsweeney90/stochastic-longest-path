#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Static scheduling main simulator module. 
"""

import numpy as np
import networkx as nx
import itertools as it
from copy import deepcopy
from networkx.drawing.nx_agraph import to_agraph
from statistics import median
from collections import defaultdict

class Task:
    """
    Represent static tasks.
    """         
    def __init__(self, task_type=None):
        """
        Create Task object.
        
        Parameters
        ------------------------
        task_type - None/string
        String identifying the name of the task, e.g., "GEMM".
        
        Attributes
        ------------------------
        type - None/string
        Initialized to task_type.
        
        ID - int
        Identification number of the Task in its DAG.
        
        entry - bool
        True if Task has no predecessors, False otherwise.
        
        exit - bool
        True if Task has no successors, False otherwise.
        
        The following 4 attributes are usually set after initialization by functions which
        take a Node object as a target platform.
        
        CPU_time - int/float
        The Task's execution time on CPU Workers. 
        
        GPU_time - int/float
        The Task's execution time on GPU Workers. 
        
        acceleration_ratio - int/float
        The ratio of the Task's execution time on CPU and GPU Workers. 
        
        comm_costs - defaultdict(dict)
        Nested dict {string identifying source and target processor types : {child ID : cost}}
        e.g., self.comm_costs["CG"][5] = 10 means that the communication cost between the Task
        and the child task with ID 5 is 10 when Task is scheduled on a CPU Worker and the child 
        is scheduled on a GPU Worker.
        
        The following 4 attributes are set once the task has actually been scheduled.
        
        AST - int/float
        The actual start time of the Task.
        
        AFT- int/float
        The actual finish time of the Task.
        
        scheduled - bool
        True if Task has been scheduled on a Worker, False otherwise.
        
        where_scheduled - None/int
        The numerical ID of the Worker that the Task has been scheduled on. Often useful.
        
        Comments
        ------------------------
        1. It would perhaps be more useful in general to take all attributes as parameters since this
           is more flexible but as we rarely work at the level of individual Tasks this isn't necessary
           for our purposes.        
        """           
         
        self.type = task_type  
        self.ID = None    
        self.entry = False 
        self.exit = False    
        
        self.comp_costs = {"C" : 0.0, "G" : 0.0} 
        self.acceleration_ratio = None  # Set when costs are set.
        self.comm_costs = {}
        for p, q in it.product(self.comp_costs, self.comp_costs):
            self.comm_costs[p + q] = {}        
        
        self.FT = None  
        self.scheduled = False  
        self.where_scheduled = None                 
    
    def reset(self):
        """Resets some attributes to defaults so execution of the task can be simulated again."""
        self.FT = None   
        self.scheduled = False
        self.where_scheduled = None      

class DAG:
    """
    Represents a task DAG.   
    """
    def __init__(self, G, name=None): 
        """
        The DAG is a collection of Tasks with a topology defined by a Networkx DiGraph object.        
        
        Parameters
        ------------------------
        name - string
        The name of the application the DAG represents, e.g., "Cholesky".
        
        Attributes
        ------------------------
        name - string
        Ditto above.
        
        DAG - DiGraph from Networkx module
        Represents the topology of the DAG.
        
        n_tasks - int
        The number of tasks in the DAG.
        
        The following attributes summarize topological information and are usually set
        by compute_topological_info when necessary.
               
        n_edges - None/int
        The number of edges in the DAG. 
        
        edge_density - None/float
        The ratio of the number of edges in the DAG to the maximum possible for a DAG with the same
        number of tasks. 
        
        CCR - dict {string : float}
        Summarizes the computation-to-communication ratio (CCR) values for different platforms in the
        form {platform name : DAG CCR}.         
        
        Comments
        ------------------------
        1. It seems a little strange to make the CCR a dict but it avoided having to compute it over 
           and over again for the same platforms in some scripts.
        """  
        
        self.name = name 
        self.graph = G
        self.n_tasks = len(G)  
        self.n_edges = G.number_of_edges()   
        self.top_sort = list(nx.topological_sort(self.graph)) 
        for i, t in enumerate(self.top_sort):
            t.ID = i
            if not list(self.graph.predecessors(t)):
                t.entry = True
            elif not list(self.graph.successors(t)):
                t.exit = True
        
    def reset(self):
        """Resets some Task attributes to defaults so scheduling of the DAG can be simulated again."""
        for task in self.graph:
            task.reset() 

    def scheduled(self):
        """Returns True all the tasks in the DAG have been scheduled, False if not."""
        return all(task.scheduled for task in self.graph)   
    
    def ready_to_schedule(self, task):
        """
        Determine if Task is ready to schedule - i.e., all precedence constraints have been 
        satisfied or it is an entry task.
        
        Parameters
        ------------------------
        dag - DAG object
        The DAG to which the Task belongs.                
                                         
        Returns
        ------------------------
        bool
        True if Task can be scheduled, False otherwise.         
        
        Notes
        ------------------------
        1. Returns False if Task has already been scheduled.
        """
        
        if task.scheduled:
            return False  
        if task.entry: 
            return True
        for parent in self.graph.predecessors(task):
            if not parent.scheduled:
                return False
        return True    
    
    def get_ready_tasks(self):
        """
        Identify the tasks that are ready to schedule.               

        Returns
        ------------------------                          
        List
        All tasks in the DAG that are ready to be scheduled.                 
        """       
        return list(t for t in self.graph if self.ready_to_schedule(t))
    
    def makespan(self, partial=False):
        """
        Compute the makespan of the DAG.
        
        Parameters
        ------------------------        
        partial - bool
        If True, only computes makespan of all tasks that have been scheduled so far, not the entire DAG. 

        Returns
        ------------------------         
        int/float
        The makespan of the (possibly incomplete) DAG.        
        """         
        if partial:
            return max(t.FT for t in self.graph if t.FT is not None)  
        return max(t.FT for t in self.graph if t.exit) 
    
    def minimal_serial_time(self):
        """
        Computes the minimum makespan of the DAG on a single Worker of the platform.
        
        Parameters
        ------------------------
        platform - Node object (see Environment.py module)
        The target platform.        

        Returns
        ------------------------                          
        float
        The minimal serial time.      
        
        Notes
        ------------------------                          
        1. Assumes all task CPU and GPU times are set.        
        """        
        return min(sum(task.comp_costs["C"] for task in self.graph), sum(task.comp_costs["G"] for task in self.graph))   
    
    def optimistic_cost_table(self, expected_comm=False, platform=None):
        """
        Incorporated into optimistic critical path method.
        """  
        
        d = {"CC" : 0, "CG" : 1, "GC" : 1, "GG" : 0}          
        OCT = defaultdict(lambda: defaultdict(float))  
        
        backward_traversal = list(reversed(self.top_sort))
        for task in backward_traversal:
            for p in ["C", "G"]:
                OCT[task.ID][p] = 0.0
                if task.exit:
                    continue
                child_values = []
                for child in self.graph.successors(task):
                    if expected_comm:
                        action_values = [OCT[child.ID][q] + d[p + q] * platform.average_comm_cost(task, child) + child.comp_costs[q] for q in ["C", "G"]]
                    else:
                        action_values = [OCT[child.ID][q] + d[p + q] * task.comm_costs[p + q][child.ID] + child.comp_costs[q] for q in ["C", "G"]]
                    child_values.append(min(action_values))
                OCT[task.ID][p] += max(child_values)      
        return OCT 
    
    def expected_cost_table(self, platform, weighted=False):
        """
        Incorporated into optimistic critical path method.
        """  
                
        u = defaultdict(lambda: defaultdict(float))  
        
        backward_traversal = list(reversed(self.top_sort))
        for task in backward_traversal:
            u[task.ID]["C"] = 0.0
            u[task.ID]["G"] = 0.0
            if task.exit:
                continue
            
            A = task.acceleration_ratio if weighted else 1
            d1 = platform.n_CPUs + A * platform.n_GPUs
            
            c_child_values, g_child_values = [], []
            for child in self.graph.successors(task):
                B = child.acceleration_ratio if weighted else 1
                d2 = platform.n_CPUs + B * platform.n_GPUs
                common = platform.n_CPUs * (u[child.ID]["C"] + child.comp_costs["C"]) 
                common += B * platform.n_GPUs * (u[child.ID]["G"] + child.comp_costs["G"])
                
                c_maximand = platform.n_CPUs * (platform.n_CPUs - 1) * task.comm_costs["CC"][child.ID]
                c_maximand += platform.n_CPUs * B * platform.n_GPUs * task.comm_costs["CG"][child.ID]
                c_maximand /= d1 
                c_maximand += common
                c_maximand /= d2
                c_child_values.append(c_maximand)
                
                g_maximand = A * platform.n_GPUs * platform.n_CPUs * task.comm_costs["GC"][child.ID]
                g_maximand += A * platform.n_GPUs * B * (platform.n_GPUs - 1) * task.comm_costs["GG"][child.ID]
                g_maximand /= d1
                g_maximand += common
                g_maximand /= d2
                g_child_values.append(g_maximand)     
            u[task.ID]["C"] += max(c_child_values) 
            u[task.ID]["G"] += max(g_child_values)
        return u
    
    def optimistic_critical_path(self, direction="downward", lookahead=False):
        """
        Computes the optimistic finish time, as defined in the Heterogeneous Optimistic Finish Time (HOFT) algorithm,
        of all tasks assuming they are scheduled on either CPU or GPU. 
        Used in the HOFT heuristic - see Heuristics.py.                  

        Returns
        ------------------------                          
        OCP - Nested defaultdict
        The optimistic finish time table in the form {Task 1: {Worker 1 : c1, Worker 2 : c2, ...}, ...}.         
        
        Notes
        ------------------------ 
        1. No target platform is necessary.
        2. If "remaining == True" is almost identical to the Optimistic Cost Table (OCT) from the PEFT heuristic.
        """  
             
        OCP = defaultdict(lambda: defaultdict(float))  
        d = {"CC" : 0, "CG" : 1, "GC" : 1, "GG" : 0} 
        
        if direction == "upward":
            backward_traversal = list(reversed(self.top_sort))
            for task in backward_traversal:
                for p in ["C", "G"]:
                    OCP[task.ID][p] = task.comp_costs[p] if not lookahead else 0.0
                    if task.exit:
                        continue
                    child_values = []
                    for child in self.graph.successors(task):
                        if lookahead:
                            action_values = [OCP[child.ID][q] + d[p + q] * task.comm_costs[p + q][child.ID] + child.comp_costs[q] for q in ["C", "G"]]
                        else:
                            action_values = [OCP[child.ID][q] + d[p + q] * task.comm_costs[p + q][child.ID] for q in ["C", "G"]]
                        child_values.append(min(action_values))
                    OCP[task.ID][p] += max(child_values)             
        else:
            for task in self.top_sort:
                for p in ["C", "G"]:
                    OCP[task.ID][p] = task.comp_costs[p]
                    if task.entry:
                        continue
                    parent_values = []
                    for parent in self.graph.predecessors(task):
                        action_values = [OCP[parent.ID][q] + d[q + p] * parent.comm_costs[q + p][task.ID] for q in ["C", "G"]]
                        parent_values.append(min(action_values))
                    OCP[task.ID][p] += max(parent_values)   
        return OCP    

    def expected_critical_path(self, platform, direction="downward", lookahead=False, weighted=False):
        """
        Similar to above but expected critical path...
        """  
                         
        if direction == "upward":
            u = defaultdict(lambda: defaultdict(float))
            backward_traversal = list(reversed(self.top_sort))
            for task in backward_traversal:
                # Compute u^c and u^g.
                u[task.ID]["C"] = task.comp_costs["C"] if not lookahead else 0.0
                u[task.ID]["G"] = task.comp_costs["G"] if not lookahead else 0.0
                if task.exit:
                    continue
                c_child_values, g_child_values = [], []
                A = task.acceleration_ratio if weighted else 1.0
                d1 = platform.n_CPUs + A * platform.n_GPUs
                for child in self.graph.successors(task):
                    B = child.acceleration_ratio if weighted else 1.0
                    d2 = platform.n_CPUs + B * platform.n_GPUs
                    if lookahead:
                        common = platform.n_CPUs * (u[child.ID]["C"] + child.comp_costs["C"]) 
                        common += B * platform.n_GPUs * (u[child.ID]["G"] + child.comp_costs["G"])
                    else:
                        common = platform.n_CPUs * u[child.ID]["C"] 
                        common += B * platform.n_GPUs * u[child.ID]["G"]
                    
                    c_maximand = platform.n_CPUs * (platform.n_CPUs - 1) * task.comm_costs["CC"][child.ID]
                    c_maximand += platform.n_CPUs * B * platform.n_GPUs * task.comm_costs["CG"][child.ID]
                    c_maximand /= d1 
                    c_maximand += common
                    c_maximand /= d2
                    c_child_values.append(c_maximand)
                    
                    g_maximand = A * platform.n_GPUs * platform.n_CPUs * task.comm_costs["GC"][child.ID]
                    g_maximand += A * platform.n_GPUs * B * (platform.n_GPUs - 1) * task.comm_costs["GG"][child.ID]
                    g_maximand /= d1
                    g_maximand += common
                    g_maximand /= d2
                    g_child_values.append(g_maximand)                    
                    
                u[task.ID]["C"] += max(c_child_values) 
                u[task.ID]["G"] += max(g_child_values) 
            return u
                
        else: 
            d = defaultdict(lambda: defaultdict(float))
            for task in self.top_sort:
                # Compute d^c and d^g.
                d[task.ID]["C"] = task.comp_costs["C"] 
                d[task.ID]["G"] = task.comp_costs["G"] 
                if task.entry:
                    continue
                c_parent_values, g_parent_values = [], []
                A = task.acceleration_ratio if weighted else 1.0
                d1 = platform.n_CPUs + A * platform.n_GPUs
                for parent in self.graph.predecessors(task):
                    B = parent.acceleration_ratio if weighted else 1.0
                    d2 = platform.n_CPUs + B * platform.n_GPUs
                    
                    common = platform.n_CPUs * d[parent.ID]["C"] 
                    common += B * platform.n_GPUs * d[parent.ID]["G"]
                    
                    c_maximand = platform.n_CPUs * (platform.n_CPUs - 1) * parent.comm_costs["CC"][task.ID]
                    c_maximand += platform.n_CPUs * B * platform.n_GPUs * parent.comm_costs["GC"][task.ID]
                    c_maximand /= d1
                    c_maximand += common
                    c_maximand /= d2
                    c_parent_values.append(c_maximand)
                    
                    g_maximand = A * platform.n_GPUs * platform.n_CPUs * parent.comm_costs["CG"][task.ID]
                    g_maximand += A * platform.n_GPUs * B * (platform.n_GPUs - 1) * parent.comm_costs["GG"][task.ID]
                    g_maximand /= d1
                    g_maximand += common
                    g_maximand /= d2
                    g_parent_values.append(g_maximand)                    
                    
                d[task.ID]["C"] += max(c_parent_values) 
                d[task.ID]["G"] += max(g_parent_values)
            return d
            
    def set_costs(self, acc_ratios, target_ccr, platform, shares=[0, 1/3, 1/3, 1/3]):
        """
        Sets computation and communication costs for randomly generated DAGs (e.g., from the STG).
        
        Parameters
        ------------------------
        platform - Node object (see Environment.py module)
        The target platform.           
        
        target_ccr - float/int
        The CCR we want the DAG to have on the target platform. Due to stochasticity in how we choose 
        communication costs this is not precise so we might need to double check afterwards.              
                      
        Notes
        ------------------------
        1. I not already set, we assume that GPU times are uniformly distributed integers between 1 and 100.
        2. We assume that CPU-CPU communication costs are zero and all others are of similar magnitude to
           one another (as we typically have throughout).
        3. Communication costs are sampled from a Gamma distribution with a computed mean and standard deviation
           to try and achieve the desired CCR value for the DAG.
        """   
        
        if isinstance(acc_ratios, tuple) or isinstance(acc_ratios, list):
            dist, mu, sigma = acc_ratios
        
        # Set the computation costs.
        for task in self.graph:
            if task.comp_costs["G"] == 0.0:
                task.comp_costs["G"] = np.random.randint(1, 100)         
            if isinstance(acc_ratios, dict) or isinstance(acc_ratios, defaultdict):
                task.acceleration_ratio = acc_ratios[task.type] 
            elif isinstance(acc_ratios, tuple) or isinstance(acc_ratios, list):
                if dist == "GAMMA" or dist == "gamma":
                    task.acceleration_ratio = np.random.gamma(shape=(mu/sigma)**2, scale=sigma**2/mu) 
                elif dist == "NORMAL" or dist == "normal":
                    task.acceleration_ratio = abs(np.random.normal(mu, sigma)) 
                else:
                    raise ValueError('Unrecognized acceleration ratio distribution specified in set_costs!')                    
            task.comp_costs["C"] = task.comp_costs["G"] * task.acceleration_ratio
        
        # Set the communication costs.        
        # Compute the expected total compute of the entire DAG.
        cpu_compute = list(task.comp_costs["C"] for task in self.graph)
        gpu_compute = list(task.comp_costs["G"] for task in self.graph)
        expected_total_compute = sum(cpu_compute) * platform.n_CPUs + sum(gpu_compute) * platform.n_GPUs
        expected_total_compute /= platform.n_workers
        
        # Calculate the expected communication cost of the entire DAG - i.e., for all edges.        
        expected_total_comm = expected_total_compute / target_ccr
        expected_comm_per_edge = expected_total_comm / self.n_edges
        for task in self.top_sort:
            for child in self.graph.successors(task):
                if isinstance(acc_ratios, tuple) or isinstance(acc_ratios, list):
                    if dist == "GAMMA" or dist == "gamma":
                        w_bar = np.random.gamma(shape=1.0, scale=expected_comm_per_edge)
                    elif dist == "NORMAL" or dist == "normal":
                        w_bar = abs(np.random.normal(expected_comm_per_edge, expected_comm_per_edge))
                else:
                    w_bar = np.random.uniform(0, 2) * expected_comm_per_edge
                # Now sets the costs according to the relative shares.
                x = w_bar * platform.n_workers**2   
                s0, s1, s2, s3 = shares
                d = s0 * platform.n_CPUs * (platform.n_CPUs - 1)
                d += (s1 + s3) * platform.n_CPUs * platform.n_GPUs
                d += s2 * platform.n_GPUs * (platform.n_GPUs - 1)
                x /= d
                task.comm_costs["CC"][child.ID] = s0 * x
                task.comm_costs["CG"][child.ID] = s1 * x
                task.comm_costs["GG"][child.ID] = s2 * x
                task.comm_costs["GC"][child.ID] = s3 * x                 
           
    def sort_by_upward_rank(self, platform, avg_type="HEFT", return_ranks=False):
        """
        Sorts all tasks in the DAG by decreasing/non-increasing order of upward rank.
        
        Parameters
        ------------------------
        platform - Node object (see Environment.py module)
        The target platform.
        
        avg_type - string
        How the tasks and edges should be weighted in platform.average_comm_cost and task.average_execution_cost.
        Default is "HEFT" which is mean values over all processors. See referenced methods for more options.
        
        return_rank_values - bool
        If True, method also returns the upward rank values for all tasks.
        
        verbose - bool
        If True, print the ordering of all tasks to the screen. Useful for debugging. 

        Returns
        ------------------------                          
        priority_list - list
        Scheduling list of all Task objects prioritized by upward rank.
        
        If return_rank_values == True:
        task_ranks - dict
        Gives the actual upward ranks of all tasks in the form {task : rank_u}.
        
        Notes
        ------------------------ 
        1. "Upward rank" is also called "bottom-level".        
        """      
        
        # Traverse the DAG starting from the exit task.
        backward_traversal = list(reversed(self.top_sort))        
        # Compute the upward rank of all tasks recursively.
        task_ranks = {}
        for t in backward_traversal:
            task_ranks[t] = platform.average_comp_cost(t, avg_type=avg_type) 
            try:
                task_ranks[t] += max(platform.average_comm_cost(parent=t, child=s, avg_type=avg_type) + task_ranks[s] for s in self.graph.successors(t))
            except ValueError:
                pass  
            # print(t.ID, task_ranks[t])
        priority_list = list(reversed(sorted(task_ranks, key=task_ranks.get)))
                    
        if return_ranks:
            return priority_list, task_ranks
        return priority_list  
    
    def sort_by_downward_rank(self, platform, avg_type="HEFT", return_ranks=False):
        """
        Sorts all tasks in the DAG by increasing/non-decreasing order of downward rank.
        
        Parameters
        ------------------------
        platform - Node object (see Environment.py module)
        The target platform.
        
        avg_type - string
        How the tasks and edges should be weighted in platform.average_comm_cost and task.average_execution_cost.
        Default is "HEFT" which is mean values over all processors. See referenced methods for more options.
        
        return_rank_values - bool
        If True, method also returns the downward rank values for all tasks.
        
        verbose - bool
        If True, print the ordering of all tasks to the screen. Useful for debugging. 

        Returns
        ------------------------                          
        priority_list - list
        Scheduling list of all Task objects prioritized by downward rank.
        
        If return_rank_values == True:
        task_ranks - dict
        Gives the actual downward ranks of all tasks in the form {task : rank_d}.
        
        Notes
        ------------------------ 
        1. "Downward rank" is also called "top-level".        
        """ 
              
        # Compute the downward rank of all tasks recursively.
        task_ranks = {}
        for t in self.top_sort:
            task_ranks[t] = 0.0
            try:
                task_ranks[t] += max(platform.average_comp_cost(p, avg_type) + platform.average_comm_cost(parent=p, child=t, avg_type=avg_type) +
                          task_ranks[p] for p in self.graph.predecessors(t))
            except ValueError:
                pass          
        priority_list = list(sorted(task_ranks, key=task_ranks.get))
                        
        if return_ranks:
            return priority_list, task_ranks
        return priority_list   
        
    def sort_by_Fulkerson_rank(self, platform, return_f=False, downward=False, weighted=False):
        """
        TODO: changed to incorporate final node cost, check still everything still works (especially downward version).
        Notes:
            1. "Original" version is as described in Fulkerson's original paper. Much slower than the default but might be wanted.
            2. The default version computes the ranks using the more computationally efficient method for computing Fulkerson's "f"
               numbers as first stated by Clingen (1964? Check) and elucidated by Elmaghraby (1967). Assumes that costs are independent
               but that's a fairly standard assumption anyway.
        """          
        
        # Define the respective probabilities of each potential edge weight.
        # edge_probs[0] == CPU-same CPU, edge_probs[1] == CPU-different CPU, edge_probs[2] == CPU-GPU,
        # edge_probs[3] == GPU-same GPU, edge_probs[4] == GPU-different GPU, edge_probs[5] == GPU-CPU.
        if not weighted:
            d = platform.n_workers**2
            edge_probs = [platform.n_CPUs/d, (platform.n_CPUs * (platform.n_CPUs - 1))/d, platform.n_CPUs * platform.n_GPUs / d,
                          platform.n_GPUs/d, (platform.n_GPUs * (platform.n_GPUs - 1))/d, platform.n_GPUs * platform.n_CPUs / d] 
        f = {}
        
        if not downward:        
            backward_traversal = list(reversed(self.top_sort))
            for t in backward_traversal:
                if t.exit:
                    f[t] = 0.0    
                    continue
                children = list(self.graph.successors(t))                  
                # Find alpha and the potential z values to check.
                alpha, Z = 0.0, []
                for c in children:  
                    if c.exit:
                        alpha = max(alpha, min(t.comp_costs["C"] + c.comp_costs["C"], t.comp_costs["G"] + c.comp_costs["G"]))
                        n = [t.comp_costs["C"] + c.comp_costs["C"],
                             t.comm_costs["CC"][c.ID] + t.comp_costs["C"] + c.comp_costs["C"],
                             t.comm_costs["CG"][c.ID] + t.comp_costs["C"] + c.comp_costs["G"], 
                             t.comp_costs["G"] + c.comp_costs["G"],
                             t.comm_costs["GG"][c.ID] + t.comp_costs["G"] + c.comp_costs["G"],
                             t.comm_costs["GC"][c.ID] + t.comp_costs["G"] + c.comp_costs["C"]]                        
                    else:
                        alpha = max(alpha, f[c] + min(t.comp_costs["C"], t.comp_costs["G"]))
                        n = [f[c] + t.comp_costs["C"],
                             f[c] + t.comm_costs["CC"][c.ID] + t.comp_costs["C"],
                             f[c] + t.comm_costs["CG"][c.ID] + t.comp_costs["C"], 
                             f[c] + t.comp_costs["G"],
                             f[c] + t.comm_costs["GG"][c.ID] + t.comp_costs["G"],
                             f[c] + t.comm_costs["GC"][c.ID] + t.comp_costs["G"]]
                    Z += n          
                # Compute f. 
                f[t] = 0.0
                Z = list(set(Z))    # TODO: might still need a check to prevent rounding errors.
                for z in Z:
                    if alpha - z > 1e-6:   
                        continue
                    # Iterate over edges and compute the two products.
                    plus, minus = 1, 1                
                    for c in children:
                        # Compute zdash = z - f_c.
                        zdash = z - f[c] 
                        # Define the edge costs.
                        if weighted:    
                            r1, r2 = t.acceleration_ratio, c.acceleration_ratio
                            d = (platform.n_CPUs + r1 * platform.n_GPUs) * (platform.n_CPUs + r2 * platform.n_GPUs)
                            edge_probs = [platform.n_CPUs/d, 
                                          (platform.n_CPUs * (platform.n_CPUs - 1))/d, 
                                          platform.n_CPUs * r2 * platform.n_GPUs / d,
                                          r1 * r2 * platform.n_GPUs/d, 
                                          (r1 * r2 * platform.n_GPUs * (platform.n_GPUs - 1))/d,
                                          r1 * platform.n_GPUs * platform.n_CPUs / d]
                        if c.exit:
                            edge_costs = [[t.comp_costs["C"] + c.comp_costs["C"], edge_probs[0]],
                              [t.comm_costs["CC"][c.ID] + t.comp_costs["C"] + c.comp_costs["C"], edge_probs[1]],
                              [t.comm_costs["CG"][c.ID] + t.comp_costs["C"] + c.comp_costs["G"], edge_probs[2]],
                              [t.comp_costs["G"] + c.comp_costs["G"], edge_probs[3]],
                              [t.comm_costs["GG"][c.ID] + t.comp_costs["G"] + c.comp_costs["G"], edge_probs[4]],
                              [t.comm_costs["GC"][c.ID] + t.comp_costs["G"] + c.comp_costs["C"], edge_probs[5]]] 
                        else:
                            edge_costs = [[t.comp_costs["C"], edge_probs[0]],
                              [t.comm_costs["CC"][c.ID] + t.comp_costs["C"], edge_probs[1]],
                              [t.comm_costs["CG"][c.ID] + t.comp_costs["C"], edge_probs[2]],
                              [t.comp_costs["G"], edge_probs[3]],
                              [t.comm_costs["GG"][c.ID] + t.comp_costs["G"], edge_probs[4]],
                              [t.comm_costs["GC"][c.ID] + t.comp_costs["G"], edge_probs[5]]]                                          
                        # Compute m and p.
                        m = sum(e[1] for e in edge_costs if zdash - e[0] > 1e-6)
                        minus *= m 
                        p = m + sum(e[1] for e in edge_costs if abs(zdash - e[0]) < 1e-6)
                        plus *= p
                    # Add to f.                                    
                    f[t] += z * (plus - minus)
                # print(t.ID, f[t])
            # Sort tasks by rank. 
            priority_list = list(reversed(sorted(f, key=f.get)))         
        else:
            for t in self.top_sort:
                if t.entry:
                    f[t] = 0.0
                    continue
                parents = list(self.graph.predecessors(t)) 
                # Find alpha and the potential z values to check.
                alpha, Z = 0.0, []
                for p in parents: 
                    if t.exit:
                        alpha = max(alpha, f[p] + min(p.comp_costs["C"] + t.comp_costs["C"], p.comp_costs["G"] + p.comp_costs["G"]))
                        n = [f[p] + p.comp_costs["C"] + t.comp_costs["C"],
                             f[p] + p.comm_costs["CC"][t.ID] + p.comp_costs["C"] + t.comp_costs["C"],
                             f[p] + p.comm_costs["CG"][t.ID] + p.comp_costs["C"] + t.comp_costs["G"], 
                             f[p] + p.comp_costs["G"] + t.comp_costs["G"],
                             f[p] + p.comm_costs["GG"][t.ID] + p.comp_costs["G"] + t.comp_costs["G"],
                             f[p] + p.comm_costs["GC"][t.ID] + p.comp_costs["G"] + t.comp_costs["C"]]
                    else:
                        alpha = max(alpha, f[p] + min(p.comp_costs["C"], p.comp_costs["G"])) 
                        n = [f[p] + p.comp_costs["C"],
                             f[p] + p.comm_costs["CC"][t.ID] + p.comp_costs["C"],
                             f[p] + p.comm_costs["CG"][t.ID] + p.comp_costs["C"], 
                             f[p] + p.comp_costs["G"],
                             f[p] + p.comm_costs["GG"][t.ID] + p.comp_costs["G"],
                             f[p] + p.comm_costs["GC"][t.ID] + p.comp_costs["G"]]
                    Z += n  
                # Compute f. 
                f[t] = 0.0
                Z = list(set(Z))    # TODO: might still need a check to prevent rounding errors.
                for z in Z:
                    if alpha - z > 1e-6:   
                        continue
                    # Iterate over edges and compute the two products.
                    plus, minus = 1, 1                
                    for p in parents:
                        # Compute zdash = z - f_p.
                        zdash = z - f[p]    
                        # Define the edge costs.
                        if weighted:    # TODO: check this.
                            r1, r2 = p.acceleration_ratio, t.acceleration_ratio
                            d = (platform.n_CPUs + r1 * platform.n_GPUs) * (platform.n_CPUs + r2 * platform.n_GPUs)
                            edge_probs = [platform.n_CPUs/d, 
                                          (platform.n_CPUs * (platform.n_CPUs - 1))/d, 
                                          platform.n_CPUs * r2 * platform.n_GPUs / d,
                                          platform.n_GPUs/d, 
                                          r1 * r2 * (platform.n_GPUs * (platform.n_GPUs - 1))/d,
                                          r1 * platform.n_GPUs * platform.n_CPUs / d]
                        if t.exit:
                            edge_costs = [[p.comp_costs["C"] + t.comp_costs["C"], edge_probs[0]],
                              [p.comm_costs["CC"][t.ID] + p.comp_costs["C"] + t.comp_costs["C"], edge_probs[1]],
                              [p.comm_costs["CG"][t.ID] + p.comp_costs["C"] + t.comp_costs["G"], edge_probs[2]],
                              [p.comp_costs["G"] + t.comp_costs["G"], edge_probs[3]],
                              [p.comm_costs["GG"][t.ID] + p.comp_costs["G"] + t.comp_costs["G"], edge_probs[4]],
                              [p.comm_costs["GC"][t.ID] + p.comp_costs["G"] + t.comp_costs["C"], edge_probs[5]]] 
                        else:
                            edge_costs = [[p.comp_costs["C"], edge_probs[0]],
                              [p.comm_costs["CC"][t.ID] + p.comp_costs["C"], edge_probs[1]],
                              [p.comm_costs["CG"][t.ID] + p.comp_costs["C"], edge_probs[2]],
                              [p.comp_costs["G"], edge_probs[3]],
                              [p.comm_costs["GG"][t.ID] + p.comp_costs["G"], edge_probs[4]],
                              [p.comm_costs["GC"][t.ID] + p.comp_costs["G"], edge_probs[5]]]                                        
                        # Compute m and p.
                        m = sum(e[1] for e in edge_costs if zdash - e[0] > 1e-6)
                        minus *= m 
                        pl = m + sum(e[1] for e in edge_costs if abs(zdash - e[0]) < 1e-6)
                        plus *= pl
                    # Add to f.                                    
                    f[t] += z * (plus - minus) 
                # print("Task: {}, f = {}".format(t.ID, f[t]))
            # Sort tasks by rank. 
            priority_list = list(sorted(f, key=f.get))
                
        if return_f:
            return priority_list, f
        return priority_list   
    
    def sort_by_preference_rank(self, weight="C", comp_func="r", ranking="upward", platform=None, return_ranks=False, table=None):
        """Preference-based rankings."""
        
        if table is not None:
            if weight[:2] == "OC":
                OCP = table
            elif weight[:2] == "EC":
                ECP = table
        elif weight[:3] == "ECT":
            if platform is None:
                raise ValueError("Called sort_by_preference_rank with weight == ECT but no platform specified!")
            d = platform.n_workers**2
        elif weight == "OCD":
            OCP = self.optimistic_critical_path(direction="downward")
        elif weight == "OCU":
            OCP = self.optimistic_critical_path(direction="upward")
        elif weight == "ECD" or weight == "ECD-B":
            w = True if weight[-1] == "B" else False
            ECP = self.expected_critical_path(platform, direction="downward", weighted=w)
        elif weight == "ECU" or weight == "ECU-B":
            w = True if weight[-1] == "B" else False
            ECP = self.expected_critical_path(platform, direction="upward", weighted=w)
            
        # Compute weights.
        a = {}
        for t in self.top_sort:
            # Compute the CPU and GPU weights.
            if weight == "OCD" or weight == "OCU":
                C, G = OCP[t.ID]["C"], OCP[t.ID]["G"]
            elif weight[:3] == "ECD" or weight[:3] == "ECU":
                C, G = ECP[t.ID]["C"], ECP[t.ID]["G"]
            else:
                C, G = t.comp_costs["C"], t.comp_costs["G"]
                children = list(self.graph.successors(t))
                if weight[:3] == "ECT":
                    ec, eg = 0.0, 0.0
                    A = t.acceleration_ratio if weight[-1] == "B" else 1.0
                    for child in children:
                        B = child.acceleration_ratio if weight[-1] == "B" else 1.0
                        ec += platform.n_CPUs * (platform.n_CPUs - 1) * t.comm_costs["CC"][child.ID]
                        ec += platform.n_CPUs * B * platform.n_GPUs * t.comm_costs["CG"][child.ID]
                        eg += A * platform.n_GPUs * platform.n_CPUs * t.comm_costs["GC"][child.ID]
                        eg += A * platform.n_GPUs * B * (platform.n_GPUs - 1) * t.comm_costs["GG"][child.ID]
                    ec /= d
                    eg /= d
                    C += ec 
                    G += eg   
                elif weight == "MCT":
                    mc, mg = 0.0, 0.0
                    for child in children:
                        mc += max(t.comm_costs["CC"][child.ID], t.comm_costs["CG"][child.ID])
                        mg += max(t.comm_costs["GC"][child.ID], t.comm_costs["GG"][child.ID])
                    C += mc
                    G += mg
                    
            # Now compute a_i.
            fastest, slowest = min(C, G), max(C, G)
            if comp_func == "r":
                a[t.ID] = slowest / fastest
            elif comp_func == "d":
                a[t.ID] = slowest - fastest
            elif comp_func == "nc":
                a[t.ID] = (slowest - fastest) / (slowest / fastest) 
        
        task_ranks = {}
        if ranking == "upward":  
            # Traverse the DAG starting from the exit task.
            backward_traversal = list(reversed(self.top_sort)) 
            for t in backward_traversal:
                task_ranks[t] = a[t.ID]
                # Add the maximum child task rank to ensure precedence constraints are met.
                try:
                    task_ranks[t] += max(task_ranks[s] for s in self.graph.successors(t))
                except ValueError:
                    pass
            priority_list = list(reversed(sorted(task_ranks, key=task_ranks.get)))
        elif ranking == "downward":
            for t in self.top_sort:
                task_ranks[t] = a[t.ID]
                try:
                    task_ranks[t] += max(task_ranks[s] for s in self.graph.predecessors(t))
                except ValueError:
                    pass
            priority_list = list(sorted(task_ranks, key=task_ranks.get))            
                               
        if return_ranks:
            return priority_list, task_ranks
        return priority_list      
                 
    def draw_graph(self, filepath):
        """
        Draws the DAG and saves the image.
        
        Parameters
        ------------------------        
        filepath - string
        Destination for image. 

        Notes
        ------------------------                           
        1. See https://stackoverflow.com/questions/39657395/how-to-draw-properly-networkx-graphs       
        """        
        G = deepcopy(self.graph)        
        G.graph['graph'] = {'rankdir':'TD'}  
        G.graph['node']={'shape':'circle', 'color':'#348ABD', 'style':'filled', 'fillcolor':'#E5E5E5', 'penwidth':'3.0'}
        G.graph['edges']={'arrowsize':'4.0', 'penwidth':'5.0'}       
        A = to_agraph(G)        
        # Add identifying colors if task types are known.
        for task in G:
            if task.type == "GEMM":
                n = A.get_node(task)  
                n.attr['color'] = 'black'
                n.attr['fillcolor'] = '#E24A33'
                n.attr['label'] = 'G'
            elif task.type == "POTRF":
                n = A.get_node(task)   
                n.attr['color'] = 'black'
                n.attr['fillcolor'] = '#348ABD'
                n.attr['label'] = 'P'
            elif task.type == "SYRK":
                n = A.get_node(task)   
                n.attr['color'] = 'black'
                n.attr['fillcolor'] = '#988ED5'
                n.attr['label'] = 'S'
            elif task.type == "TRSM":
                n = A.get_node(task)    
                n.attr['color'] = 'black'
                n.attr['fillcolor'] = '#FBC15E'
                n.attr['label'] = 'T' 
        A.layout('dot')
        A.draw('{}/{}_{}tasks_DAG.png'.format(filepath, self.name.split(" ")[0], self.n_tasks)) 
    
    def print_info(self, platforms=None, return_mst_and_cp=False, detailed=False, filepath=None):
        """
        Print basic information about the DAG, either to screen or as txt file.
        
        Parameters
        ------------------------
        platforms - None/Node object (see Environment.py module)/list
        Compute more specific information about the DAG when executed on the platform (if Node)
        or multiple platforms (if list of Nodes).
        
        detailed - bool
        If True, print information about individual Tasks.
        
        filepath - string
        Destination for txt file.                           
        """
        
        print("--------------------------------------------------------", file=filepath)
        print("DAG INFO", file=filepath)
        print("--------------------------------------------------------", file=filepath)   
        print("Name: {}".format(self.name), file=filepath)
        
        # Basic information.
        print("Number of tasks: {}".format(self.n_tasks), file=filepath)
        print("Number of edges: {}".format(self.n_edges), file=filepath)
        max_edges = (self.n_tasks * (self.n_tasks - 1)) / 2 
        edge_density = self.n_edges / max_edges 
        print("Edge density: {}".format(edge_density), file=filepath)
                
        # Cost information.
        cpu_costs = list(task.comp_costs["C"] for task in self.graph)
        gpu_costs = list(task.comp_costs["G"] for task in self.graph)
        acc_ratios = list(task.acceleration_ratio for task in self.graph)
        cpu_mu, cpu_sigma = np.mean(cpu_costs), np.std(cpu_costs)
        print("Mean task CPU cost: {}, standard deviation: {}".format(cpu_mu, cpu_sigma), file=filepath)
        gpu_mu, gpu_sigma = np.mean(gpu_costs), np.std(gpu_costs)
        print("Mean task GPU cost: {}, standard deviation: {}".format(gpu_mu, gpu_sigma), file=filepath)            
        acc_mu, acc_sigma = np.mean(acc_ratios), np.std(acc_ratios)
        print("Mean task acceleration ratio: {}, standard deviation: {}".format(acc_mu, acc_sigma), file=filepath)   
        mst = self.minimal_serial_time()
        print("Minimal serial time: {}".format(mst), file=filepath)
        OCP = self.optimistic_critical_path()
        cp = max(min(OCP[task.ID][p] for p in OCP[task.ID]) for task in self.graph if task.exit) 
        print("Optimal critical path length: {}".format(cp), file=filepath)
            
        if isinstance(platforms, list):
            for p in platforms:
                task_mu = (p.n_GPUs * gpu_mu + p.n_CPUs * cpu_mu) / p.n_workers
                print("\nMean task cost on {} platform: {}".format(p.name, task_mu), file=filepath)
                ccr = p.CCR(self)
                print("Computation-to-communication ratio on {} platform: {}".format(p.name, ccr), file=filepath)        
                    
        if detailed:
            print("\n--------------------------------------------------------", file=filepath) 
            print("DETAILED BREAKDOWN OF TASKS IN DAG:", file=filepath)
            print("--------------------------------------------------------", file=filepath) 
            for task in self.graph:
                print("\nTask ID: {}".format(task.ID), file=filepath)
                if task.entry:
                    print("Entry task.", file=filepath)
                if task.exit:
                    print("Exit task.", file=filepath)
                if task.type is not None:
                    print("Task type: {}".format(task.type), file=filepath) 
                print("CPU cost: {}".format(task.comp_costs["C"]), file=filepath)
                print("GPU cost: {}".format(task.comp_costs["G"]), file=filepath)
                print("Acceleration ratio: {}".format(task.acceleration_ratio), file=filepath)               
        print("--------------------------------------------------------", file=filepath) 
        
        if return_mst_and_cp:
            return mst, cp
          
class Worker:
    """
    Represents any CPU or GPU processing resource. 
    """
    def __init__(self, worker_type="C", ID=None):
        """
        Create the Worker object.
        
        Parameters
        --------------------
        GPU - bool
        True if Worker is a GPU. Assumed to be a CPU unless specified otherwise.
        
        ID - Int
        Assigns an integer ID to the task. Often very useful.        
        """        
        
        self.type = worker_type  
        self.ID = ID   
        self.load = []  # Tasks scheduled on the processor.
        self.idle = True    # True if no tasks currently scheduled on the processor. 
                
    def earliest_finish_time(self, task, dag, platform, insertion=True):
        """
        Returns the estimated earliest start time for a Task on the Worker.
        
        Parameters
        ------------------------
        task - Task object (see Graph.py module)
        Represents a (static) task.
        
        dag - DAG object (see Graph.py module)
        The DAG to which the task belongs.
              
        platform - Node object
        The Node object to which the Worker belongs.
        Needed for calculating communication costs.
        
        insertion - bool
        If True, use insertion-based scheduling policy - i.e., task can be scheduled 
        between two already scheduled tasks, if permitted by dependencies.
        
        Returns
        ------------------------
        float 
        The earliest finish time for task on Worker.        
        """    
        
        task_cost = task.comp_costs[self.type] 
        
        # If no tasks scheduled on processor...
        if self.idle:   
            if task.entry: 
                return (task_cost, 0)
            else:
                return (task_cost + max(p.FT + platform.comm_cost(p, task, p.where_scheduled, self.ID) for p in dag.graph.predecessors(task)), 0)  
                # TODO: ideally want to remove the FT attribute for tasks. 
            
        # At least one task already scheduled on processor... 
                
        # Find earliest time all task predecessors have finished and the task can theoretically start.     
        drt = 0
        if not task.entry:                    
            parents = dag.graph.predecessors(task) 
            drt += max(p.FT + platform.comm_cost(p, task, p.where_scheduled, self.ID) for p in parents)  # TODO: ideally want to remove this.
        
        if not insertion:
            return (task_cost + max(self.load[-1][2], drt), -1)
        
        # Check if it can be scheduled before any other task in the load.
        prev_finish_time = 0.0
        for i, t in enumerate(self.load):
            if t[1] < drt:
                prev_finish_time = t[2]
                continue
            poss_finish_time = max(prev_finish_time, drt) + task_cost
            if poss_finish_time <= t[1]:
                return (poss_finish_time, i) # TODO: should this be i - 1?
            prev_finish_time = t[2]
        
        # No valid gap found.
        return (task_cost + max(self.load[-1][2], drt), -1)    
        
    def schedule_task(self, task, finish_time=None, load_idx=None, dag=None, platform=None, insertion=True):
        """
        Schedules the task on the Worker.
        
        Parameters
        ------------------------
        task - Task object (see Graph.py module)
        Represents a (static) task.
                
        dag - DAG object (see Graph.py module)
        The DAG to which the task belongs.
              
        platform - Node object
        The Node object to which the Worker belongs. 
        Needed for calculating communication costs, although this is a bit unconventional.
        
        insertion - bool
        If True, use insertion-based scheduling policy - i.e., task can be scheduled 
        between two already scheduled tasks, if permitted by dependencies.
        
        start_time - float
        If not None, schedules task at this start time. Validity is checked with 
        valid_start_time which raises ValueError if it fails. Should be used very carefully!
        
        finish_time - float
        If not None, taken to be task's actual finish time. 
        Should be used with great care (see note below!)
        
        Notes
        ------------------------
        1. If finish_time, doesn't check that all task predecessors have actually been scheduled.
           This is so we can do lookahead in e.g., platform.estimate_finish_times and to save repeated
           calculations in some circumstances but should be used very, very carefully!                 
        """         
                        
        # Set task attributes.
        if finish_time is None:
            finish_time, load_idx = self.earliest_finish_time(task, dag, platform, insertion=insertion) 
        
        start_time = finish_time - task.comp_costs[self.type] 
        
        # Add to load.           
        if self.idle or not insertion or load_idx < 0:             
            self.load.append((task.ID, start_time, finish_time, task.type))  
            if self.idle:
                self.idle = False
        else: 
            self.load.insert(load_idx, (task.ID, start_time, finish_time, task.type))                
        
        # Set the task attributes.
        task.FT = finish_time 
        task.scheduled = True
        task.where_scheduled = self.ID  

    def unschedule_task(self, task):
        """
        Unschedules the task on the Worker.
        
        Parameters
        ------------------------
        task - Task object (see Graph.py module)
        Represents a (static) task.                 
        """
        
        # Remove task from the load.
        for t in self.load:
            if t[0] == task.ID:
                self.load.remove(t)
                break
        # Revert Worker to idle if necessary.
        if not len(self.load):
            self.idle = True
        # Reset the task itself.    
        task.reset()                         
        
    def print_schedule(self, filepath=None):
        """
        Print the current tasks scheduled on the Worker, either to screen or as txt file.
        
        Parameters
        ------------------------
        filepath - string
        Destination for schedule txt file.                           
        """        
        print("WORKER {}, TYPE: {}PU: ".format(self.ID, self.type), file=filepath)
        for t in self.load:
            type_info = " Task type: {},".format(t[3]) if t[3] is not None else ""
            print("Task ID: {},{} Start time = {}, finish time = {}.".format(t[0], type_info, t[1], t[2]), file=filepath)  

class Platform:
    """          
    A Node is basically just a collection of CPU and GPU Worker objects.
    """
    def __init__(self, CPUs, GPUs, name=None):
        """
        Initialize the Node by giving the number of CPUs and GPUs.
        
        Parameters
        ------------------------
        CPUs - int
        The number of CPUs.

        GPUs - int
        The number of GPUs.
        
        name - string
        An identifying name for the Node. Often useful.
        """
        
        self.name = name     
        self.n_CPUs, self.n_GPUs = CPUs, GPUs 
        self.n_workers = self.n_CPUs + self.n_GPUs      # Often useful.
        self.workers = []       # List of all Worker objects.
        for i in range(self.n_CPUs):
            self.workers.append(Worker(ID=i))          
        for j in range(self.n_GPUs):
            self.workers.append(Worker(worker_type="G", ID=self.n_CPUs + j))         
    
    def reset(self):
        """Resets some attributes to defaults so we can simulate the execution of another DAG."""
        for w in self.workers:
            w.load = []   
            w.idle = True                    
    
    def comm_cost(self, parent, child, source_id, target_id):   
        """
        Compute the communication time from a parent task to a child.
        
        Parameters
        ------------------------
        parent - Task object (see Graph.py module)
        The parent task that is sending its data.
        
        child - Task object (see Graph.py module)
        The child task that is receiving data.
        
        source_id - int
        The ID of the Worker on which parent is scheduled.
        
        target_id - int
        The ID of the Worker on which child may be scheduled.
        
        Returns
        ------------------------
        float 
        The communication time between parent and child.        
        """       
        
        if source_id == target_id:
            return 0.0         
        source_type = self.workers[source_id].type
        target_type = self.workers[target_id].type     
        return parent.comm_costs[source_type + target_type][child.ID]  
    
    def average_comp_cost(self, task, avg_type="HEFT", children=None):
        """
        Compute the "average" computation time of the Task. 
        Usually used for setting priorities in HEFT and similar heuristics.
        
        Parameters
        ------------------------
        platform - Node object (see Environment.py module)
        The target platform.
                
        avg_type - string
        How the average should be computed. 
        Options:
            - "HEFT", use mean values over all processors as in HEFT.
            - "median", use median values over all processors. 
            - "worst", always use largest possible computation cost.
            - "simple worst", always use largest possible computation cost.
            - "best", always use smallest possible computation cost.
            - "simple best", always use smallest possible computation cost.
            - "HEFT-WM", compute mean over all processors, weighted by acceleration ratio.
            - "PS", processor speedup. Cost = max(CPU cost, GPU cost) / min(CPU cost, GPU cost).
            - "D", difference. Cost = max(CPU cost, GPU cost) - min(CPU cost, GPU cost).
            - "SFB". Cost = ( max(CPU cost, GPU cost) - min(CPU cost, GPU cost) ) / ( max(CPU cost, GPU cost) / min(CPU cost, GPU cost) ). 
                                         
        Returns
        ------------------------
        float 
        The average computation cost of the Task. 
        
        Notes
        ------------------------
        1. "median", "worst", "simple worst", "best", "simple best" were all considered by Zhao and Sakellariou (2003). 
        2. "PS", "D" and "SFB" are from Shetti, Fahmy and Bretschneider (2013).
        """
        
        C, G = task.comp_costs["C"], task.comp_costs["G"]            
        
        if avg_type == "HEFT" or avg_type == "mean" or avg_type == "MEAN" or avg_type == "M":
            return (C * self.n_CPUs + G * self.n_GPUs) / self.n_workers
        elif avg_type == "median" or avg_type == "MEDIAN":
            costs = [C] * self.n_CPUs + [G] * self.n_GPUs
            return median(costs)
        elif avg_type == "worst" or avg_type == "W" or avg_type == "simple worst" or avg_type == "SW":
            return max(C, G)
        elif avg_type == "best" or avg_type == "B" or avg_type == "simple best" or avg_type == "sb":
            return min(C, G)   
        elif avg_type == "HEFT-WM" or avg_type == "WM":
            r = task.acceleration_ratio
            return (C * self.n_CPUs + r * G * self.n_GPUs) / (self.n_CPUs + r * self.n_GPUs)   
            
        raise ValueError('No avg_type, e.g., "mean" or "median", specified for average_execution_cost.')                 
    
    def average_comm_cost(self, parent, child, avg_type="HEFT"): 
        """
        Compute the "average" communication time from parent to child tasks. 
        Usually used for setting priorities in HEFT and similar heuristics.
        
        Parameters
        ------------------------
        parent - Task object (see Graph.py module)
        The parent task that is sending its data.
        
        child - Task object (see Graph.py module)
        The child task that is receiving data.
        
        avg_type - string
        How the average should be computed. 
        Options:
            - "HEFT", use mean values over all processors as in HEFT.
            - "median", use median values over all processors. 
            - "worst", assume each task is on its slowest processor type and compute corresponding communication cost.
            - "simple worst", always use largest possible communication cost.
            - "best", assume each task is on its fastest processor type and compute corresponding communication cost.
            - "simple best", always use smallest possible communication cost.
            - "HEFT-WM", compute mean over all processors, weighted by task acceleration ratios.
            - "PS", "D", "SFB" - speedup-based avg_types from Shetti, Fahmy and Bretschneider (2013). 
               Returns zero in all three cases so definitions can be found in average_execution_cost
               method in the Task class in Graph.py.
                                         
        Returns
        ------------------------
        float 
        The average communication cost between parent and child. 
        
        Notes
        ------------------------
        1. "median", "worst", "simple worst", "best", "simple best" were all considered by Zhao and Sakellariou (2003). 
        """
                
        if avg_type == "HEFT" or avg_type == "mean" or avg_type == "MEAN" or avg_type == "M":            
            c_bar = self.n_CPUs * (self.n_CPUs - 1) * parent.comm_costs["CC"][child.ID] 
            c_bar += self.n_CPUs * self.n_GPUs * parent.comm_costs["CG"][child.ID]
            c_bar += self.n_CPUs * self.n_GPUs * parent.comm_costs["GC"][child.ID]
            c_bar += self.n_GPUs * (self.n_GPUs - 1) * parent.comm_costs["GG"][child.ID]
            c_bar /= (self.n_workers**2)
            return c_bar            
            
        elif avg_type == "median" or avg_type == "MEDIAN":
            costs = self.n_CPUs * (self.n_CPUs - 1) * [parent.comm_costs["CC"][child.ID]] 
            costs += self.n_CPUs * self.n_GPUs * [parent.comm_costs["CG"][child.ID]]
            costs += self.n_CPUs * self.n_GPUs * [parent.comm_costs["GC"][child.ID]]
            costs += self.n_GPUs * (self.n_GPUs - 1) * [parent.comm_costs["GG"][child.ID]]
            costs += self.n_workers * [0]
            return median(costs)
        
        elif avg_type == "worst" or avg_type == "WORST":
            parent_worst_proc = "C" if parent.comp_costs["C"] > parent.comp_costs["G"] else "G"
            child_worst_proc = "C" if child.comp_costs["C"] > child.comp_costs["G"] else "G"
            if parent_worst_proc == "C" and child_worst_proc == "C" and self.n_CPUs == 1:
                return 0.0
            if parent_worst_proc == "G" and child_worst_proc == "G" and self.n_GPUs == 1:
                return 0.0
            return parent.comm_costs[parent_worst_proc + child_worst_proc][child.ID]
        
        elif avg_type == "simple worst" or avg_type == "SW":
            return max(parent.comm_costs["CC"][child.ID], parent.comm_costs["CG"][child.ID], parent.comm_costs["GC"][child.ID], parent.comm_costs["GG"][child.ID])
        
        elif avg_type == "best" or avg_type == "BEST":
            parent_best_proc = "G" if parent.comp_costs["C"] > parent.comp_costs["G"] else "C"
            child_best_proc = "G" if child.comp_costs["C"] > child.comp_costs["G"] else "C"
            if parent_best_proc == child_best_proc:
                return 0.0
            return parent.comm_costs[parent_best_proc + child_best_proc][child.ID]
        
        elif avg_type == "simple best" or avg_type == "sb":
            return min(parent.comm_costs["CC"][child.ID], parent.comm_costs["CG"][child.ID], parent.comm_costs["GC"][child.ID], parent.comm_costs["GG"][child.ID])         
                
        elif avg_type == "HEFT-WM" or avg_type == "WM":
            A, B = parent.acceleration_ratio, child.acceleration_ratio
            c_bar = self.n_CPUs * (self.n_CPUs - 1) * parent.comm_costs["CC"][child.ID] 
            c_bar += self.n_CPUs * B * self.n_GPUs * parent.comm_costs["CG"][child.ID]
            c_bar += A * self.n_GPUs * self.n_CPUs * parent.comm_costs["GC"][child.ID]
            c_bar += A * self.n_GPUs * B * (self.n_GPUs - 1) * parent.comm_costs["GG"][child.ID]
            c_bar /= ((self.n_CPUs + A * self.n_GPUs) * (self.n_CPUs + B * self.n_GPUs))
            return c_bar           
        
        raise ValueError('No avg_type (e.g., "mean" or "median") specified for average_comm_cost.')          
        
    def CCR(self, dag, avg_type="HEFT"):
        """
        Compute and set the computation-to-communication ratio (CCR) for the DAG on the 
        target platform.          
        """
        
        exp_comm, exp_comp = 0.0, 0.0
        for task in dag.top_sort:
            exp_comp += self.average_comp_cost(task, avg_type=avg_type)
            children = dag.graph.successors(task)
            for child in children:
                exp_comm += self.average_comm_cost(task, child, avg_type=avg_type)
        return exp_comp / exp_comm                
    
    def print_info(self, print_schedule=False, filepath=None):
        """
        Print basic information about the Platform, either to screen or as txt file.
        
        Parameters
        ------------------------
        filepath - string
        Destination for txt file.                           
        """        
        print("----------------------------------------------------------------------------------------------------------------", file=filepath)
        print("PLATFORM INFO", file=filepath)
        print("----------------------------------------------------------------------------------------------------------------", file=filepath)
        print("Name: {}".format(self.name), file=filepath)
        print("{} CPUs, {} GPUs".format(self.n_CPUs, self.n_GPUs), file=filepath)
        print("----------------------------------------------------------------------------------------------------------------\n", file=filepath)  
        
        if print_schedule:
            print("----------------------------------------------------------------------------------------------------------------", file=filepath)
            print("CURRENT SCHEDULE", file=filepath)
            print("----------------------------------------------------------------------------------------------------------------", file=filepath)
            for w in self.workers:
                w.print_schedule(filepath=filepath)  
            mkspan = max(w.load[-1][2] for w in self.workers if w.load) 
            print("\nMAKESPAN: {}".format(mkspan), file=filepath)            
            print("----------------------------------------------------------------------------------------------------------------\n", file=filepath)
    
    def summarize_schedule(self, dag, schedule):
        """
        Follow the input schedule.
        """        
        
        info = {}     
        # Compute makespan.
        for task in schedule:
            p = schedule[task]
            self.workers[p].schedule_task(task, dag=dag, platform=self)  
        mkspan = dag.makespan() 
        info["MAKESPAN"] = mkspan
        # Reset DAG and platform.
        dag.reset()
        self.reset() 
        
        # Compute CCR and critical path of the fixed-cost DAG.
        backward_traversal = list(reversed(dag.top_sort))  
        cp_lengths, total_comp, total_comm = {}, 0.0, 0.0
        for task in backward_traversal:
            w_t = task.comp_costs["C"] if schedule[task] < self.n_CPUs else task.comp_costs["G"] 
            total_comp += w_t
            cp_lengths[task.ID] = w_t 
            children = list(dag.graph.successors(task))
            source = "C" if schedule[task] < self.n_CPUs else "G"
            maximand = 0.0
            for c in children:
                target = "C" if schedule[c] < self.n_CPUs else "G"
                edge_cost = task.comm_costs[source + target][c.ID] if schedule[task] != schedule[c] else 0.0
                total_comm += edge_cost
                maximand = max(maximand, edge_cost + cp_lengths[c.ID])
            cp_lengths[task.ID] += maximand
        
        ccr = total_comp / total_comm
        info["CCR"] = ccr        
        cp = cp_lengths[dag.top_sort[0].ID]
        info["CRITICAL PATH"] = cp
        slr = mkspan / cp
        info["SCHEDULE LENGTH RATIO"] = slr
        
        return info
        
            
# =============================================================================
# Heuristics.   
# =============================================================================            
            
def HEFT(dag, platform, priority_list=None, avg_type="HEFT", return_schedule=False, schedule_dest=None):
    """
    Heterogeneous Earliest Finish Time.
    'Performance-effective and low-complexity task scheduling for heterogeneous computing',
    Topcuoglu, Hariri and Wu, 2002.
    
    Parameters
    ------------------------    
    dag - DAG object (see Graph.py module)
    Represents the task DAG to be scheduled.
          
    platform - Node object (see Environment.py module)
    Represents the target platform.  
    
    priority_list - None/list
    If not None, an ordered list which gives the order in which tasks are to be scheduled. 
    
    avg_type - string
    How the tasks and edges should be weighted in dag.sort_by_upward_rank.
    Default is "HEFT" which is mean values over all processors as in the original paper. 
    See platform.average_comm_cost and platform.average_execution_cost for other options.
    
    return_schedule - bool
    If True, return the schedule computed by the heuristic.
             
    schedule_dest - None/string
    Path to save schedule. 
    
    Returns
    ------------------------
    mkspan - float
    The makespan of the schedule produced by the heuristic.
    
    If return_schedule == True:
    pi - dict
    The schedule in the form {task : ID of Worker it is scheduled on}.    
    """ 
    
    if return_schedule:
        pi = {}
    
    # List all tasks by upward rank unless alternative is specified.
    if priority_list is None:
        priority_list = dag.sort_by_upward_rank(platform, avg_type=avg_type)   
    
    # Schedule the tasks.
    for t in priority_list:    
        
        # Compute the finish time on all workers and identify the fastest (with ties broken consistently by np.argmin).   
        worker_finish_times = list(w.earliest_finish_time(t, dag, platform) for w in platform.workers)
        min_val = min(worker_finish_times, key=lambda w:w[0]) 
        min_worker = worker_finish_times.index(min_val)                       
        
        # Schedule the task on the chosen worker. 
        ft, idx = min_val
        platform.workers[min_worker].schedule_task(t, finish_time=ft, load_idx=idx)        
        if return_schedule:
            pi[t] = min_worker
                    
    # If schedule_dest, print the schedule to file.
    if schedule_dest is not None: 
        print("The tasks were scheduled in the following order:", file=schedule_dest)
        for t in priority_list:
            print(t.ID, file=schedule_dest)
        print("\n", file=schedule_dest)
        platform.print_info(print_schedule=True, filepath=schedule_dest)
        
    # Compute makespan.
    mkspan = dag.makespan() 
    
    # Reset DAG and platform.
    dag.reset()
    platform.reset() 
    
    if return_schedule:
        return mkspan, pi    
    return mkspan 

def PEFT(dag, platform, return_schedule=False, schedule_dest=None, expected_comm=True):
    """
    Predict Earliest Finish Time.
    'List scheduling algorithm for heterogeneous systems by an optimistic cost table',
    Arabnejad and Barbosa, 2014.
    
    Parameters
    ------------------------    
    dag - DAG object (see Graph.py module)
    Represents the task DAG to be scheduled.
          
    platform - Node object (see Environment.py module)
    Represents the target platform. 
    
    priority_list - None/list
    If not None, an ordered list which gives the order in which tasks are to be scheduled. 
        
    return_schedule - bool
    If True, return the schedule computed by the heuristic.
             
    schedule_dest - None/string
    Path to save schedule. 
    
    Returns
    ------------------------
    mkspan - float
    The makespan of the schedule produced by the heuristic.
    
    If return_schedule == True:
    pi - defaultdict(int)
    The schedule in the form {Task : ID of Worker it is scheduled on}.    
    """ 
    
    if return_schedule or schedule_dest is not None:
        pi = {}
                
    if expected_comm:    
        OCT = dag.optimistic_cost_table(expected_comm=expected_comm, platform=platform)
    else:
        OCT = dag.optimistic_cost_table()       
    
    task_ranks = {t : (platform.n_CPUs * OCT[t.ID]["C"] + platform.n_GPUs * OCT[t.ID]["G"]) / platform.n_workers for t in dag.top_sort} 
    
    ready_tasks = list(t for t in dag.top_sort if t.entry)    
    while len(ready_tasks):   
        # Find ready task with highest priority (ties broken randomly according to max function).
        t = max(ready_tasks, key=task_ranks.get) 
        # Find fastest CPU and GPU workers for t.
        worker_finish_times = list(w.earliest_finish_time(t, dag, platform) for w in platform.workers)
        min_cpu_val = min(worker_finish_times[:platform.n_CPUs], key=lambda w:w[0]) 
        min_cpu = worker_finish_times.index(min_cpu_val)
        min_gpu_val = min(worker_finish_times[platform.n_CPUs:], key=lambda w:w[0]) 
        min_gpu = worker_finish_times[platform.n_CPUs:].index(min_gpu_val) + platform.n_CPUs 
        # Add optimistic critical path length to finish times and compare.
        if min_cpu_val[0] + OCT[t.ID]["C"] < min_gpu_val[0] + OCT[t.ID]["G"]:
            min_worker = min_cpu
            ft, idx = min_cpu_val
        else:
            min_worker = min_gpu
            ft, idx = min_gpu_val
        # Schedule the task.
        platform.workers[min_worker].schedule_task(t, finish_time=ft, load_idx=idx)          
        if return_schedule or schedule_dest is not None:
            pi[t] = min_worker 
        # Update ready tasks.                          
        ready_tasks.remove(t)
        for c in dag.graph.successors(t):
            if dag.ready_to_schedule(c):
                ready_tasks.append(c) 
        
    # If schedule_dest, print the schedule to file.
    if schedule_dest is not None: 
        print("The tasks were scheduled in the following order:", file=schedule_dest)
        for t in pi:
            print(t.ID, file=schedule_dest)
        print("\n", file=schedule_dest)
        platform.print_info(print_schedule=True, filepath=schedule_dest)

    # Compute makespan.
    mkspan = dag.makespan()        
    
    # Reset DAG and platform.
    dag.reset()
    platform.reset()  
      
    if return_schedule:
        return mkspan, pi    
    return mkspan 

def CPOP(dag, platform, return_schedule=False, schedule_dest=None):
    """
    Critical-Path-on-a-Processor (CPOP).
    'Performance-effective and low-complexity task scheduling for heterogeneous computing',
    Topcuoglu, Hariri and Wu, 2002.
    """  
    
    if return_schedule or schedule_dest is not None:
        pi = {} 
    
    # Compute upward and downward ranks of all tasks to find priorities.
    _, upward_ranks = dag.sort_by_upward_rank(platform, return_ranks=True)
    _, downward_ranks = dag.sort_by_downward_rank(platform, return_ranks=True)
    task_ranks = {t.ID : upward_ranks[t] + downward_ranks[t] for t in dag.graph}     
    
    # Identify the tasks on the critical path.
    ready_tasks = list(t for t in dag.graph if t.entry)  
    cp_tasks = set()
    for t in ready_tasks:
        if any(abs(task_ranks[s.ID] - task_ranks[t.ID]) < 1e-6 for s in dag.graph.successors(t)):
            cp = t
            cp_prio = task_ranks[t.ID] 
            cpu_cost, gpu_cost = t.comp_costs["C"], t.comp_costs["G"]
            cp_tasks.add(cp.ID)
            break        
    while not cp.exit:
        cp = np.random.choice(list(s for s in dag.graph.successors(cp) if abs(task_ranks[s.ID] - cp_prio) < 1e-6))
        cp_tasks.add(cp.ID)
        cpu_cost += cp.comp_costs["C"]
        gpu_cost += cp.comp_costs["G"]
    # Find the fastest worker for the CP tasks.
    cp_worker = platform.workers[0] if cpu_cost < gpu_cost else platform.workers[platform.n_CPUs]     
       
    while len(ready_tasks):
        t = max(ready_tasks, key=lambda t : task_ranks[t.ID])
        
        if t.ID in cp_tasks:
            cp_worker.schedule_task(t, dag=dag, platform=platform)
            if return_schedule or schedule_dest is not None:
                pi[t] = cp_worker.ID
        else:
            worker_finish_times = list(w.earliest_finish_time(t, dag, platform) for w in platform.workers)
            min_val = min(worker_finish_times, key=lambda w:w[0]) 
            min_worker = worker_finish_times.index(min_val) 
            ft, idx = min_val
            platform.workers[min_worker].schedule_task(t, finish_time=ft, load_idx=idx)        
            if return_schedule or schedule_dest is not None:
                pi[t] = min_worker
    
        # Update ready tasks.                          
        ready_tasks.remove(t)
        for c in dag.graph.successors(t):
            if dag.ready_to_schedule(c):
                ready_tasks.append(c)       
    
    # If schedule_dest, save the priority list and schedule.
    if schedule_dest is not None: 
        print("The tasks were scheduled in the following order:", file=schedule_dest)
        for t in pi:
            print(t.ID, file=schedule_dest)
        print("\n", file=schedule_dest)
        platform.print_info(print_schedule=True, filepath=schedule_dest)       
    
    # Compute makespan.
    mkspan = dag.makespan()        
    
    # Reset DAG and platform.
    dag.reset()
    platform.reset()  
      
    if return_schedule:
        return mkspan, pi    
    return mkspan 

def HEFT_NC(dag, platform, threshold=0.3, return_schedule=False, schedule_dest=None):
    """
    HEFT No Cross (HEFT-NC).
    'Optimization of the HEFT algorithm for a CPU-GPU environment,'
    Shetti, Fahmy and Bretschneider (2013).
    """
    
    if return_schedule or schedule_dest is not None:
        pi = {}
    
    # Compute all tasks weights.
    _, task_ranks = dag.sort_by_preference_rank(comp_func="nc", return_ranks=True)           
        
    ready_tasks = list(t for t in dag.graph if t.entry)    
    while len(ready_tasks):          
        t = max(ready_tasks, key=task_ranks.get)          
        worker_finish_times = list(w.earliest_finish_time(t, dag, platform) for w in platform.workers)
        min_val = min(worker_finish_times, key=lambda w:w[0]) 
        min_worker = worker_finish_times.index(min_val)        
        
        w = t.comp_costs["C"] if min_worker < platform.n_CPUs else t.comp_costs["G"]
        if abs(w - min(t.comp_costs["C"], t.comp_costs["G"])) < 1e-6:
            ft, idx = min_val
            platform.workers[min_worker].schedule_task(t, finish_time=ft, load_idx=idx)        
            if return_schedule or schedule_dest is not None:
                pi[t] = min_worker
        else:
            ft_min, idx_min = min_val
            if min_worker < platform.n_CPUs:
                fast_val = min(worker_finish_times[platform.n_CPUs:], key=lambda w:w[0])
                fast_worker = worker_finish_times[platform.n_CPUs:].index(fast_val) + platform.n_CPUs
            else:
                fast_val = min(worker_finish_times[:platform.n_CPUs], key=lambda w:w[0]) 
                fast_worker = worker_finish_times.index(fast_val)
            ft_fast, idx_fast = fast_val
            max_i = max(t.comp_costs["C"], t.comp_costs["G"])
            min_i = min(t.comp_costs["C"], t.comp_costs["G"])
            w_i = abs(max_i - min_i) / (max_i / min_i)
            w_abs = abs(ft_min - ft_fast) / (ft_min / ft_fast)
            if w_i / w_abs <= threshold:
                platform.workers[min_worker].schedule_task(t, finish_time=ft_min, load_idx=idx_min) 
                if return_schedule or schedule_dest is not None:
                    pi[t] = min_worker
            else:
                platform.workers[fast_worker].schedule_task(t, finish_time=ft_fast, load_idx=idx_fast)
                if return_schedule or schedule_dest is not None:
                    pi[t] = fast_worker
                              
        # Update ready tasks.                          
        ready_tasks.remove(t)
        for c in dag.graph.successors(t):
            if dag.ready_to_schedule(c):
                ready_tasks.append(c)
            
    # If schedule_dest, save the priority list and schedule.
    if schedule_dest is not None: 
        print("The tasks were scheduled in the following order:", file=schedule_dest)
        for t in pi:
            print(t.ID, file=schedule_dest)
        print("\n", file=schedule_dest)
        platform.print_info(print_schedule=True, filepath=schedule_dest)       
    
    # Compute makespan.
    mkspan = dag.makespan()        
    
    # Reset DAG and platform.
    dag.reset()
    platform.reset()  
      
    if return_schedule:
        return mkspan, pi    
    return mkspan 

def EEFT(dag, platform, weighted=False, return_schedule=False, schedule_dest=None):
    """
    Modification of PEFT that uses HEFT-like critical path estimates in the lookahead.
    """ 
    
    if return_schedule or schedule_dest is not None:
        pi = {}
    
    U = dag.expected_cost_table(platform, weighted=weighted)    
    if weighted:
        task_ranks = {t : (platform.n_CPUs * U[t.ID]["C"] + t.acceleration_ratio * platform.n_GPUs * U[t.ID]["G"]) / (platform.n_CPUs + t.acceleration_ratio * platform.n_GPUs) for t in dag.top_sort}
    else:
        task_ranks = {t : (platform.n_CPUs * U[t.ID]["C"] + platform.n_GPUs * U[t.ID]["G"]) / (platform.n_CPUs + platform.n_GPUs) for t in dag.top_sort}
        
    ready_tasks = list(t for t in dag.top_sort if t.entry)    
    while len(ready_tasks):   
        # Find ready task with highest priority (ties broken randomly according to max function).
        t = max(ready_tasks, key=task_ranks.get) 
        # print(t.ID, task_ranks[t])
        # Find fastest CPU and GPU workers for t.
        worker_finish_times = list(w.earliest_finish_time(t, dag, platform) for w in platform.workers)
        min_cpu_val = min(worker_finish_times[:platform.n_CPUs], key=lambda w:w[0]) 
        min_cpu = worker_finish_times.index(min_cpu_val)
        min_gpu_val = min(worker_finish_times[platform.n_CPUs:], key=lambda w:w[0]) 
        min_gpu = worker_finish_times[platform.n_CPUs:].index(min_gpu_val) + platform.n_CPUs 
        # Add optimistic critical path length to finish times and compare.
        if min_cpu_val[0] + U[t.ID]["C"] < min_gpu_val[0] + U[t.ID]["G"]:
            min_worker = min_cpu
            ft, idx = min_cpu_val
        else:
            min_worker = min_gpu
            ft, idx = min_gpu_val
        # Schedule the task.
        platform.workers[min_worker].schedule_task(t, finish_time=ft, load_idx=idx)          
        if return_schedule or schedule_dest is not None:
            pi[t] = min_worker 
        # Update ready tasks.                          
        ready_tasks.remove(t)
        for c in dag.graph.successors(t):
            if dag.ready_to_schedule(c):
                ready_tasks.append(c) 
        
    # If schedule_dest, print the schedule to file.
    if schedule_dest is not None: 
        print("The tasks were scheduled in the following order:", file=schedule_dest)
        for t in pi:
            print(t.ID, file=schedule_dest)
        print("\n", file=schedule_dest)
        platform.print_info(print_schedule=True, filepath=schedule_dest)

    # Compute makespan.
    mkspan = dag.makespan()        
    
    # Reset DAG and platform.
    dag.reset()
    platform.reset()  
      
    if return_schedule:
        return mkspan, pi    
    return mkspan 

def HOFT(dag, platform, table=None, priority_list=None, return_schedule=False, schedule_dest=None):
    """
    Heterogeneous Optimistic Finish Time (HOFT).    
    """ 
    
    pi = {}
    
    # Compute OFT table if necessary.
    # OCP = table if table is not None else dag.optimistic_critical_path(direction="downward") 
    
    # Compute the priority list if not input.
    if priority_list is None:
        priority_list = dag.sort_by_preference_rank(weight="OCD", table=table)      
              
    for t in priority_list:                
        # Find fastest CPU and GPU workers for t.
        worker_finish_times = list(w.earliest_finish_time(t, dag, platform) for w in platform.workers)
        min_cpu_val = min(worker_finish_times[:platform.n_CPUs], key=lambda w:w[0]) 
        min_cpu = worker_finish_times.index(min_cpu_val)
        min_gpu_val = min(worker_finish_times[platform.n_CPUs:], key=lambda w:w[0]) 
        min_gpu = worker_finish_times[platform.n_CPUs:].index(min_gpu_val) + platform.n_CPUs          
          
        
        if (min_cpu_val[0] < min_gpu_val[0]) and (t.comp_costs["C"] < t.comp_costs["G"]):
            min_worker = min_cpu
            ft, idx = min_cpu_val
        elif (min_gpu_val[0] < min_cpu_val[0]) and (t.comp_costs["G"] < t.comp_costs["C"]):
            min_worker = min_gpu
            ft, idx = min_gpu_val
        else:  
            sc, sg = 0, 0
            for k in dag.graph.successors(t):
                c = (platform.n_CPUs - 1) * t.comm_costs["CC"][k.ID]
                c += k.acceleration_ratio * platform.n_GPUs * t.comm_costs["CG"][k.ID]
                c /= (platform.n_CPUs + k.acceleration_ratio * platform.n_GPUs)
                sc += c            
                g = platform.n_CPUs * t.comm_costs["GC"][k.ID]
                g += k.acceleration_ratio * (platform.n_GPUs - 1) * t.comm_costs["GG"][k.ID]
                g /= (platform.n_CPUs + k.acceleration_ratio * platform.n_GPUs)
                sg += g  
            # for k in dag.graph.successors(t):
            #     kp = "C" if OCP[k.ID]["C"] < OCP[k.ID]["G"] else "G"
            #     dc = t.comm_costs["C" + kp][k.ID] 
            #     dg = t.comm_costs["G" + kp][k.ID] 
            #     sc = max(sc, k.comp_costs[kp] + dc)
            #     sg = max(sg, k.comp_costs[kp] + dg) 
            # Check condition.
            if min_cpu_val[0] + sc < min_gpu_val[0] + sg:
                min_worker = min_cpu
                ft, idx = min_cpu_val
            else:
                min_worker = min_gpu
                ft, idx = min_gpu_val
            
        # Schedule the task.
        platform.workers[min_worker].schedule_task(t, finish_time=ft, load_idx=idx)          
        if return_schedule:
            pi[t] = min_worker 
                       
    # If schedule_dest, print the schedule to file.
    if schedule_dest is not None: 
        print("The tasks were scheduled in the following order:", file=schedule_dest)
        for t in priority_list:
            print(t.ID, file=schedule_dest)
        print("\n", file=schedule_dest)
        platform.print_info(print_schedule=True, filepath=schedule_dest)
        
    # Compute makespan.
    mkspan = dag.makespan() 
    
    # Reset DAG and platform.
    dag.reset()
    platform.reset() 
    
    if return_schedule:
        return mkspan, pi    
    return mkspan  