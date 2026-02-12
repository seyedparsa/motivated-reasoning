import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

load_dotenv()

from thoughts.multiple_choice import generate_responses, train_probes, evaluate_responses, evaluate_probes, evaluate_llm

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Evaluate language models on various tasks")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str)
    parser.add_argument("--reason_first", action='store_true', help="Use Chain of Thought generation")
    parser.add_argument("--bias", type=str, default=None, help="Bias to apply (e.g., 'expert', 'self')")
    parser.add_argument("--hint_idx", type=int, default=0, help="Index of the hint to provide")
    parser.add_argument("--generate", action='store_true', help="Generate outputs")
    parser.add_argument("--evaluate", action='store_true', help="Evaluate generated outputs")
    parser.add_argument("--train_probes", action='store_true', help="Train probes")
    parser.add_argument("--evaluate_probes", action='store_true', help="Evaluate probes")
    parser.add_argument("--evaluate_llm", action='store_true', help="Evaluate LLM baseline for has-switched detection")
    parser.add_argument("--probe", type=str, help="probe type (e.g., 'hint-recovery', 'mot_vs_alg', 'mot_vs_res', 'mot_vs_oth)")
    parser.add_argument("--llm", type=str, default='gpt-5-nano', help="LLM model for baseline evaluation (e.g., 'gpt-5-nano')")
    # parser.add_argument("--per_layer", action='store_true', help="Train a probe per layer")
    parser.add_argument("--universal", action='store_true', help="Use a universal probe")
    parser.add_argument("--balanced", action='store_true', help="Use balanced examples for probing")
    parser.add_argument("--filter_mentions", type=bool, default=True, help="Filter out examples whose CoT mentions the hint keyword")
    parser.add_argument("--aggregate_layers", type=str, default=None, help="Aggregate probe predictions across layers per step (all, first:K, last:K, first_last)")
    parser.add_argument("--aggregate_steps", type=str, default=None, help="Aggregate probe predictions across steps per layer (all, first:K, last:K, first_last)")

    # parser.add_argument("--n_gen", type=int,  help="Number of responses to generate")
    parser.add_argument("--bs_gen", type=int, default=64, help="Batch size for generation")
    # parser.add_argument("--n_train", type=int, help="Number of responses to train on")
    parser.add_argument("--n_questions", type=int, help="Number of questions to load")
    parser.add_argument("--n_test_questions", type=int, help="Number of questions to test on")
    parser.add_argument("--bs_probe", type=int, default=32, help="Batch size for probing")
    parser.add_argument("--n_ckpts", type=int, help="Number of samples from each response")
    parser.add_argument("--ckpt", type=str, default="rel", help="Checkpointing strategy (rel, prefix, suffix)")
    parser.add_argument("--tag", type=str, default='', help="Run tag for separating experiments (e.g., 'debug', 'ablation-v2')")
    # parser.add_argument("--n_test", type=int,  help="Number of responses to test on")    
    args = parser.parse_args()
    split = args.split or ('train' if args.dataset in ['aqua', 'commonsense_qa'] else 'test')
    reason_first = args.reason_first or (args.bias in ['expert', 'metadata'])

    if args.dataset == 'mmlu':    
        args.n_questions = 3200
        args.n_test_questions = 800
    elif args.dataset == 'aqua':
        args.n_questions = 3200
        args.n_test_questions = 800
    elif args.dataset == 'commonsense_qa':
        args.n_questions = 3200
        args.n_test_questions = 800
    elif args.dataset == 'arc-challenge':
        args.n_questions = 800
        args.n_test_questions = 200

    args.filter_mentions = False if args.probe == 'will-switch' else True

    if args.generate:
        generate_responses(args.model, args.dataset, split, reason_first, args.bias, args.hint_idx, args.n_questions, args.bs_gen, tag=args.tag)
    if args.evaluate:
        evaluate_responses(args.model, args.dataset, split)
    if args.train_probes:
        train_probes(args.model, args.dataset, split, args.n_questions, args.bias, args.probe, args.n_ckpts, args.ckpt, args.universal, args.balanced, filter_mentions=args.filter_mentions, batch_size=args.bs_probe, tag=args.tag)
    if args.evaluate_probes:
        evaluate_probes(args.model, args.dataset, split, args.n_questions, args.n_test_questions, args.bias, args.probe, args.n_ckpts, args.ckpt, args.universal, args.balanced, filter_mentions=args.filter_mentions, batch_size=args.bs_probe, aggregate_layers=args.aggregate_layers, aggregate_steps=args.aggregate_steps, tag=args.tag)
    if args.evaluate_llm:
        evaluate_llm(args.model, args.dataset, split, args.n_questions, args.n_test_questions, args.bias, args.probe, llm=args.llm, balanced=args.balanced, filter_mentions=args.filter_mentions, tag=args.tag)

