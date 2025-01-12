import numpy as np
import scipy.stats
import torch
import warnings

import flows_non_iterative as flows
import utils_non_iterative_new as utils 



class TrainingSimple:
    def __init__(self, q1_or_q2, q, args):
        assert args.tail_integral_d in [1], "d should be 1"
        self.d = args.tail_integral_d

        self.q_ratio = 'q_ratio'
        
        self.q = q

        self.args = args

        self.init_distributions()

    def init_distributions(self):
        d = self.d

        # This is used as p(x) and for the p(y|x) = N(x, 1) => y = x + N(0, 1)
        # torch.distributions.Independent(p, 1) reinterprets the first batch dim of a distribution p as an event dim.
        self.standard_normal = torch.distributions.Independent(
            torch.distributions.Normal(torch.zeros(d), torch.ones(d)), 1)

        # self.q_prime_x is the proposal distribution for training the proposal q1, see Eq. 22 in the AMCI paper
        if d == 1:
            self.p_x = self.standard_normal 
        
            self.q_prime_x = torch.distributions.Independent(
                torch.distributions.Normal(torch.ones(d) * 6, torch.sqrt(torch.tensor(0.5)) * torch.ones(d)), 1)


    @staticmethod
    def get_proposal_model(args):
        """
        Returns the proposal distribution q.
        """

        d = args.tail_integral_d

        # specification of the flow
        if args.q_ratio == 'q_ratio':
            hyper_net_in_dim = 2 * d
            flow_layers = args.layers_q_ratio
        else:
            raise Exception(f'Unrecognized value of the argument q_ratio: {args.q_ratio}.')

        flow_modules = []
        if d == 1:
            flow_modules += [flows.RadialFlow(d) for _ in range(flow_layers)]
            if args.q_ratio == 'q_ratio':
                flow_modules += [flows.OffsetFlow(d)]
            base_distribution = torch.distributions.Independent(
                torch.distributions.Normal(torch.zeros(d), torch.ones(d)), 1)

            density_esimator = flows.FlowDensityEstimator(base_distribution, flow_modules)

            # the first dimension is the batch dimension that should be 1 so we skip it in calculating the total number
            #  of parameters in the model
            parameters_nelement = sum(np.prod(param.shape[1:]) for param in density_esimator.parameters())

            hyper_net = utils.get_flow_hyper_net(args.hidden_units_per_layer,
                                                      parameters_nelement,
                                                      hyper_net_in_dim)

            q = flows.ConditionedDensityEstimator(hyper_net=hyper_net, density_estimator=density_esimator)
        else:
            raise Exception(f'Unrecognized value of the argument tail_integral_d: {args.tail_integral_d}.')

        return q

    @staticmethod
    def f(x, theta):
        return torch.minimum(torch.tensor(15000.0), torch.maximum(torch.tensor(0.0), 50*(x-theta)**5))

    def generate_dataset(self, dataset_size):
        """
        Generates a dataset of a given size.
        :param dataset_size: int, size of the dataset to be generated
        :return: the dataset in a form of a dict.
        """
        if self.q_ratio == 'q_ratio':
            
            #give me vector of 4s of size dataset_size
            theta = (torch.ones(dataset_size) * 4.0).unsqueeze(-1)

            x = self.q_prime_x.sample((dataset_size,))  
            
            y = x + self.standard_normal.sample((dataset_size,))

            # we will denote the evaluated values of the function as f_x
            f_x = self.f(x, theta)

            return {'x': x, 'y': y, 'f_x': f_x, 'theta': theta}

    def loss(self, x=None, y=None, theta=None, f_x=None):
        """
        Computes the loss function.
        """
        
        loss = -((self.p_x.log_prob(x-5) - self.q_prime_x.log_prob(x)).exp() *
                    torch.sqrt(f_x) * self.q.log_prob(x, conditioned_on=(y, theta)))   
                                    
        return loss
    
class Evaluation(TrainingSimple):
    def __init__(self, args, load_checkpoints=True):
        
        # This is used as p(x) and for the p(y|x) = N(x, 1) => y = x + N(0, 1)
        d = args.tail_integral_d
        self.d = d

        self.args = args

        self.init_distributions()

        if load_checkpoints:
            model_parameters_names = ('max_theta',)
            self.loaded_checkpoint_parameters = \
                utils.load_proposals_from_checkpoints(self, ('q_ratio'), model_parameters_names)

    def logw_for_groundtruth(self, N, y_theta):
        y, theta = (y_theta[k] for k in ['y', 'theta'])

        x = self.q_prime_x.sample((N,)).unsqueeze(0)
        x = x_minus_theta + theta.unsqueeze(1)
        # all of the samples x yield f(x)==1 by construction (because we add theta)
        #  so we don't need to call the function f
        logw_numerator = self.standard_normal.log_prob(y.unsqueeze(1) - x) + \
                         self.p_x.log_prob(x) - \
                         self.q_prime_x.log_prob(x_minus_theta)

        x = self.p_x.sample((N,))
        logw_denominator = self.standard_normal.log_prob(y.unsqueeze(1) - x.unsqueeze(0))

        return logw_numerator, logw_denominator

    def generate_ys_thetas(self, dataset_size):
        """
        Generates a dataset of a given size.
        :param dataset_size: int, size of the dataset to be generated
        :return: the dataset in a form of a dict.
        """
        #theta = self.p_theta.sample((dataset_size,))
        theta = (torch.ones(dataset_size) * 4.0).unsqueeze(-1) #shape (100,1) correct
        x = self.p_x.sample((dataset_size,)) + 5*torch.ones(dataset_size).unsqueeze(-1) #shape (100,1) correct
        y = x + self.standard_normal.sample((dataset_size,)) #we want shape (100,1)
        return {'y': y, 'theta': theta}

    def p_xy_log_prob(self, x, y):
        return self.p_x.log_prob(x-5) + self.standard_normal.log_prob(y - x)

    def samples_for_evaluation_helper(self, q_ratio, num_of_samples, y_theta):
        y, theta = (y_theta[k] for k in ['y', 'theta'])

        if q_ratio == 'q_ratio':
            conditioned_on = (y, theta)
            q = self.q_ratio
        else:
            raise Exception(f'Unrecognized value of the argument q_ratio: {q_ratio}.')

        x_tensor_samples, q_x_log_prob = q.sample(num_of_samples,
                                        conditioned_on=conditioned_on,
                                        return_logprobs=True)

        f_x_samples = self.f(x_tensor_samples, theta)
        f_x_samples = torch.sqrt(f_x_samples)  # calculate square root of f_x_samples

        p_xy_log_prob = self.p_xy_log_prob(x_tensor_samples, y)
        log_w_q = p_xy_log_prob - q_x_log_prob

        x_samples = {'x': x_tensor_samples}

        return x_tensor_samples, x_samples, f_x_samples, log_w_q, q_x_log_prob


    def samples_for_evaluation(self, num_of_samples, y_theta, device='cpu'):
        # In 'q_x_log_prob_q_ratio_x1': 'q_ratio' stands for log probs evaluated using distribution q_ratio,
        #  and 'x1' stands for 'samples from distribution q_ratio'
        x_tensor_samples, x_samples_q_ratio, f_x_samples_q_ratio, log_w_q_ratio, q_x_log_prob_q_ratio_x1 = \
            self.samples_for_evaluation_helper('q_ratio', num_of_samples, y_theta)

        # In 'q_x_log_prob_q_ratio_x2': 'q_ratio' stands for log probs evaluated using distribution q_ratio,
        #  and 'x2' stands for 'samples from distribution q_ratio'
        y, theta = (y_theta[k] for k in ['y', 'theta'])
        q_x_log_prob_q_ratio_x1 = self.q_ratio.log_prob(x_samples_q_ratio['x'], conditioned_on=(y,theta))
        q_x_log_prob_q_ratio_x2 = self.q_ratio.log_prob(x_samples_q_ratio['x'], conditioned_on=(y,theta))

        output_dict_q_ratio = {
            'x_samples_q_ratio': x_tensor_samples,
            'f_x_samples_q_ratio': f_x_samples_q_ratio,
            'log_w_q_ratio': log_w_q_ratio,
            'q_x_log_prob_q_ratio_x1': q_x_log_prob_q_ratio_x1,
            'q_x_log_prob_q_ratio_x2': q_x_log_prob_q_ratio_x2,
        }

        output_dict_q_ratio = {k: v.to(device) for k, v in output_dict_q_ratio.items()}

        return output_dict_q_ratio

    def load_ys_thetas_and_groundtruths(self, dataset_size):
        if self.d == 1:
            ys_thetas = self.generate_ys_thetas(dataset_size)
            ys, thetas = (ys_thetas[k] for k in ('y', 'theta'))
            # This variant with pytorch doesn't provide sufficient numerical precision
            #  ground_truths = 1. - torch.distributions.Normal(ys / 2., 1. / np.sqrt(2.)).cdf(thetas)
            # The line below is coming from the analytical solution.
            ground_truths = 1. - scipy.stats.norm.cdf(thetas.to('cpu').numpy().astype(np.float64),
                                                        loc=ys.to('cpu').numpy().astype(np.float64)/2.,
                                                        scale=1./np.sqrt(2.))
            if (ground_truths == 0.).any():
                warnings.warn("Ground truth should never be 0, numerical issues.")
            return ys_thetas, ground_truths