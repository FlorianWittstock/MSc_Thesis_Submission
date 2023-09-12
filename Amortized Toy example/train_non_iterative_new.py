import os.path
from collections import defaultdict
from types import SimpleNamespace

import torch

import utils_non_iterative_new as utils
import flows_non_iterative as flows


def train(args, _run, _writer):
    model = utils.get_model(args)

    q = model.TrainingSimple.get_proposal_model(args)

    optimizer = torch.optim.Adam(q.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           patience=args.scheduler_patience,
                                                           factor=args.scheduler_factor,
                                                           verbose=True)

    training = model.TrainingSimple(args.q_ratio, q, args)

    logs = defaultdict(list)

    def save_checkpoint():
        print('Saving model and logs')
        torch.save((q.state_dict(), args, logs),
                   os.path.join(args.output_folder, 'checkpoint.pytorch'))


    dataset_size = args.number_train_samples + args.number_validation_samples
    num_batches = float(args.number_train_samples) / args.minibatch_size

    for epoch_no in range(args.epochs):
        data = training.generate_dataset(dataset_size)
        train_data = {k: v[:args.number_train_samples] for k, v in data.items()}
        validation_data = {k: v[args.number_train_samples:] for k, v in data.items()}

        missteps = 0

        with torch.no_grad():
            validation_loss = training.loss(**validation_data).mean().item()

        for local_iter in range(args.max_dataset_iterations):
            print(f'validation_loss: {validation_loss:.5f}')

            train_loss = 0.
            for batch_data in utils.iterate_minibatches(args.minibatch_size, train_data):
                optimizer.zero_grad()
                loss = training.loss(**batch_data).mean()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() / num_batches

            with torch.no_grad():
                next_validation_loss = training.loss(**validation_data).mean().item()

            if next_validation_loss > validation_loss:
                missteps += 1
            validation_loss = next_validation_loss
            if missteps > args.misstep_tolerance:
                break

        scheduler.step(validation_loss)

        # Logging
        if args.loss_print != 0 and epoch_no % args.loss_print == 0:
            print(f'{epoch_no} train_loss: {train_loss:.5f}  '
                  f'validation_loss: {validation_loss:.5f}  '
                  f'local_iter: {local_iter}')

    # Save the final checkpoint at the end of training
    save_checkpoint()
    pass

config_non_iterative = {
    "checkpoint_frequency_in_seconds": 60.0,
    "checkpoint_q_ratio": "/Users/florianwittstock/Documents/ms_thesis_example/TARIS/Amortized/Non-Iterative/tail_integral_1d_q_ratio_230803_1928_529750_bb53d8b/checkpoint.pytorch",
    "epochs": 50,
    "figure_3_log2_max_samples": 15,
    "figure_3_number_of_tries": 1000,
    "figure_remse_number_of_tries": 10,
    "figure_remse_number_of_y_theta_samples": 10,
    "figure_remse_plot_snis_bound": True,
    "figure_remse_points_to_be_displayed_with_a_log_scale": 10,
    "figure_remse_xaxis_max_samples": "1e2",
    "figure_remse_ylim_lower": "1e-8",
    "figure_remse_ylim_upper": "1e6",
    "hidden_units_per_layer": 200,
    "layers_q_ratio": 10,
    "learning_rate": 0.001,
    "logs_root": None,
    "loss_print": 1,
    "max_dataset_iterations": 30,
    "max_theta": 5.0,
    "minibatch_size": 15000,
    "misstep_tolerance": 2,
    "no_cuda": 0,
    "number_of_samples_gpu_capacity": "2e6",
    "number_train_samples": 150000,
    "number_validation_samples": 15000,
    "problem_name": "tail_integral_1d",  #we keep this problem name to not change too much in the code
    "require_clean_repo": False,
    "scheduler_factor": 0.49,
    "scheduler_patience": 50,
    "tail_integral_d": 1,           #we keep this to not change too much in the code
    "output_folder": "/Users/florianwittstock/Documents/ms_thesis_example/TARIS/Amortized/New_Experiment",
}

def main(seed,
         config,
         q_ratio,
         factor):
    args = SimpleNamespace(**config)
    args.seed = seed
    args.q_ratio = q_ratio
    args.factor = factor
    
    print("Loaded configuration:")   
    print(args)  # Add this line to print the loaded configuration
    
    train(args, None, None)
    

if __name__ == '__main__':
    main(seed=0,config=config_non_iterative,q_ratio='q_ratio',factor=1.0)