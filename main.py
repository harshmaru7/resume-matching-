#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from src.matching_engine import ResumeMatchingEngine, MatchResult
from src.evaluator import Evaluator, load_evaluation_labels


def read_file(filepath: str) -> str:
    """Read text content from a file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def load_resumes_from_directory(directory: str) -> Dict[str, str]:
    """Load all text files from a directory as resumes."""
    resumes = {}
    dir_path = Path(directory)

    for file_path in dir_path.glob('*.txt'):
        resume_id = file_path.stem
        resumes[resume_id] = read_file(str(file_path))

    return resumes


def print_match_result(result: MatchResult, console: Optional['Console'] = None):
    """Print a single match result."""
    if RICH_AVAILABLE and console:
        score_color = (
            "green" if result.final_score >= 0.7 else
            "yellow" if result.final_score >= 0.4 else
            "red"
        )

        content = f"""
[bold]Resume ID:[/bold] {result.resume_id}
[bold]Final Score:[/bold] [{score_color}]{result.final_score:.3f}[/{score_color}]

[bold]Score Breakdown:[/bold]
  • Semantic Similarity: {result.semantic_score:.3f}
  • Skill Match: {result.skill_match_score:.3f}
  • Experience Alignment: {result.experience_score:.3f}

[bold]Matched Skills:[/bold] {', '.join(result.matched_skills[:10]) if result.matched_skills else 'None'}
[bold]Missing Skills:[/bold] {', '.join(result.missing_skills[:5]) if result.missing_skills else 'None'}
        """

        console.print(Panel(content, title="Match Result", border_style=score_color))
    else:
        print(f"\n{'='*50}")
        print(f"Resume ID: {result.resume_id}")
        print(f"Final Score: {result.final_score:.3f}")
        print(f"  - Semantic: {result.semantic_score:.3f}")
        print(f"  - Skills: {result.skill_match_score:.3f}")
        print(f"  - Experience: {result.experience_score:.3f}")
        print(f"Matched Skills: {', '.join(result.matched_skills[:10])}")
        print(f"Missing Skills: {', '.join(result.missing_skills[:5])}")
        print(f"{'='*50}\n")


def print_ranking_table(ranked_results, console: Optional['Console'] = None):
    if RICH_AVAILABLE and console:
        table = Table(title="Resume Ranking Results", show_header=True, header_style="bold magenta")
        table.add_column("Rank", style="cyan", justify="center")
        table.add_column("Resume ID", style="white")
        table.add_column("Score", justify="center")
        table.add_column("Semantic", justify="center")
        table.add_column("Skills", justify="center")
        table.add_column("Key Matches", style="dim")

        for result in ranked_results:
            mr = result.match_result
            score_color = (
                "green" if mr.final_score >= 0.7 else
                "yellow" if mr.final_score >= 0.4 else
                "red"
            )

            table.add_row(
                str(result.rank),
                result.resume_id,
                f"[{score_color}]{mr.final_score:.3f}[/{score_color}]",
                f"{mr.semantic_score:.3f}",
                f"{mr.skill_match_score:.3f}",
                ', '.join(mr.matched_skills[:3]) if mr.matched_skills else '-'
            )

        console.print(table)
    else:
        print("\n" + "="*80)
        print("RESUME RANKING RESULTS")
        print("="*80)
        print(f"{'Rank':<6} {'Resume ID':<35} {'Score':<8} {'Semantic':<10} {'Skills':<8}")
        print("-"*80)

        for result in ranked_results:
            mr = result.match_result
            print(f"{result.rank:<6} {result.resume_id:<35} {mr.final_score:<8.3f} {mr.semantic_score:<10.3f} {mr.skill_match_score:<8.3f}")

        print("="*80 + "\n")


def cmd_match(args):
    console = Console() if RICH_AVAILABLE else None
    if console:
        console.print("[bold blue]Resume Matching Engine[/bold blue]")
        console.print("Loading model and scoring resume...\n")
    else:
        print("Resume Matching Engine")
        print("Loading model and scoring resume...\n")

    job_description = read_file(args.jd)
    resume = read_file(args.resume)
    resume_id = Path(args.resume).stem

    engine = ResumeMatchingEngine()
    result = engine.score_resume(job_description, resume, resume_id)

    print_match_result(result, console)

    return result


def cmd_rank(args):
    console = Console() if RICH_AVAILABLE else None

    if console:
        console.print("[bold blue]Resume Matching Engine - Ranking Mode[/bold blue]")
        console.print(f"Loading resumes from: {args.resume_dir}\n")
    else:
        print("Resume Matching Engine - Ranking Mode")
        print(f"Loading resumes from: {args.resume_dir}\n")

    job_description = read_file(args.jd)
    resumes = load_resumes_from_directory(args.resume_dir)

    if not resumes:
        print(f"No resume files found in {args.resume_dir}")
        return []

    if console:
        console.print(f"Found {len(resumes)} resumes. Scoring...\n")
    else:
        print(f"Found {len(resumes)} resumes. Scoring...\n")

    engine = ResumeMatchingEngine()
    ranked_results = engine.rank_resumes(job_description, resumes)

    print_ranking_table(ranked_results, console)

    if args.output:
        output_data = [
            {
                'rank': r.rank,
                'resume_id': r.resume_id,
                'final_score': r.match_result.final_score,
                'semantic_score': r.match_result.semantic_score,
                'skill_match_score': r.match_result.skill_match_score,
                'matched_skills': r.match_result.matched_skills,
                'missing_skills': r.match_result.missing_skills,
            }
            for r in ranked_results
        ]
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to: {args.output}")

    return ranked_results


def cmd_evaluate(args):
    console = Console() if RICH_AVAILABLE else None
    if console:
        console.print("[bold blue]Resume Matching Engine - Evaluation Mode[/bold blue]\n")
    else:
        print("Resume Matching Engine - Evaluation Mode\n")
    job_description = read_file(args.jd)
    resumes = load_resumes_from_directory(args.resume_dir)
    with open(args.labels, 'r') as f:
        labels_data = json.load(f)

    labels = {}
    for resume_id, label_info in labels_data.items():
        if isinstance(label_info, dict):
            labels[resume_id] = label_info.get('score', 0.0)
        else:
            labels[resume_id] = float(label_info)

    print(f"Loaded {len(resumes)} resumes and {len(labels)} labels")

    engine = ResumeMatchingEngine()
    ranked_results = engine.rank_resumes(job_description, resumes)

    print_ranking_table(ranked_results, console)

    evaluator = Evaluator()
    eval_result = evaluator.evaluate_ranked_results(ranked_results, labels)

    print("\n")
    evaluator.print_evaluation_report(eval_result)

    if console:
        table = Table(title="Predicted vs Actual Scores", show_header=True)
        table.add_column("Resume ID", style="white")
        table.add_column("Predicted", justify="center")
        table.add_column("Actual", justify="center")
        table.add_column("Difference", justify="center")

        for result in ranked_results:
            resume_id = result.resume_id
            predicted = result.match_result.final_score
            actual = labels.get(resume_id, 0.0)
            diff = predicted - actual

            diff_color = "green" if abs(diff) < 0.2 else "yellow" if abs(diff) < 0.4 else "red"

            table.add_row(
                resume_id,
                f"{predicted:.3f}",
                f"{actual:.3f}",
                f"[{diff_color}]{diff:+.3f}[/{diff_color}]"
            )

        console.print(table)

    return eval_result


def cmd_demo(args):
    """Run a demo with the sample data."""
    console = Console() if RICH_AVAILABLE else None

    if console:
        console.print(Panel(
            "[bold]Resume Matching Engine Demo[/bold]\n\n"
            "This demo will:\n"
            "1. Load the sample job description\n"
            "2. Score all sample resumes\n"
            "3. Rank and evaluate results",
            title="Demo Mode",
            border_style="blue"
        ))
    else:
        print("\n" + "="*50)
        print("RESUME MATCHING ENGINE DEMO")
        print("="*50 + "\n")

    base_path = Path(__file__).parent
    jd_file = base_path / "data" / "job_descriptions" / "senior_ai_engineer.txt"
    resumes_folder = base_path / "data" / "resumes"
    labels_file = base_path / "data" / "evaluation_labels.json"

    if not jd_file.exists():
        print(f"Error: Job description not found at {jd_file}")
        return

    if not resumes_folder.exists():
        print(f"Error: Resume directory not found at {resumes_folder}")
        return

    class DemoArgs:
        jd = str(jd_file)
        resume_dir = str(resumes_folder)
        labels = str(labels_file)
        output = None
    cmd_evaluate(DemoArgs())


def main():
    parser = argparse.ArgumentParser(
        description="Resume Matching Engine - AI-powered resume scoring and ranking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Score a single resume
  python main.py match --jd job.txt --resume candidate.txt

  # Rank multiple resumes
  python main.py rank --jd job.txt --resume-dir ./resumes/

  # Evaluate against labeled data
  python main.py evaluate --jd job.txt --resume-dir ./resumes/ --labels labels.json

  # Run demo with sample data
  python main.py demo
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    match_parser = subparsers.add_parser('match', help='Score a single resume')
    match_parser.add_argument('--jd', required=True, help='Path to job description file')
    match_parser.add_argument('--resume', required=True, help='Path to resume file')

    rank_parser = subparsers.add_parser('rank', help='Rank multiple resumes')
    rank_parser.add_argument('--jd', required=True, help='Path to job description file')
    rank_parser.add_argument('--resume-dir', required=True, help='Directory containing resume files')
    rank_parser.add_argument('--output', '-o', help='Output JSON file for results')

    eval_parser = subparsers.add_parser('evaluate', help='Evaluate against labeled data')
    eval_parser.add_argument('--jd', required=True, help='Path to job description file')
    eval_parser.add_argument('--resume-dir', required=True, help='Directory containing resume files')
    eval_parser.add_argument('--labels', required=True, help='Path to labels JSON file')

    demo_parser = subparsers.add_parser('demo', help='Run demo with sample data')

    args = parser.parse_args()

    if args.command == 'match':
        cmd_match(args)
    elif args.command == 'rank':
        cmd_rank(args)
    elif args.command == 'evaluate':
        cmd_evaluate(args)
    elif args.command == 'demo':
        cmd_demo(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
