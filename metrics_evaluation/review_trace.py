import argparse
import sys
import os
import glob
import textwrap

# Add current directory to path to import parsers
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent_parsers.openhands_parser import parse_openhands_trajectory
from agent_parsers.sweagent_parser import parse_sweagent_trajectory
from agent_parsers.metagpt_parser import parse_metagpt_trajectory
from agent_parsers.live_sweagent_parser import parse_live_sweagent_trajectory

# ANSI Colors
GREEN = '\033[92m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
RED = '\033[91m'
CYAN = '\033[96m'
GREY = '\033[90m'
RESET = '\033[0m'
BOLD = '\033[1m'

def print_step(step):
    print(f"{GREY}--- Step {step.step_id} ---" + RESET)
    
    if step.type == 'instruction':
        print(f"{BLUE}{BOLD}USER INSTRUCTION:{RESET}")
        print(textwrap.indent(textwrap.fill(step.content, width=100), '  '))
        
    elif step.type == 'thought':
        print(f"{YELLOW}{BOLD}THOUGHT:{RESET}")
        # Thoughts can be long, but usually we want to read them fully
        print(textwrap.indent(step.content, '  '))
        
    elif step.type == 'action':
        action_type = step.metadata.get('action_type', 'unknown')
        print(f"{GREEN}{BOLD}ACTION ({action_type}):{RESET}")
        print(textwrap.indent(step.content, '  '))
        
    elif step.type == 'observation':
        content = step.content
        limit = 500
        truncated = False
        if len(content) > limit:
            content = content[:limit] + f"\n... [Truncated {len(content)-limit} chars] ..."
            truncated = True
        
        print(f"{CYAN}OBSERVATION:{RESET}")
        print(textwrap.indent(content, '  '))
        if truncated:
             print(f"{GREY}(Full output hidden for readability){RESET}")
             
    print("")

def find_log_file(agent, task_id, logs_root):
    # Normalize inputs
    agent = agent.lower()
    task_id = task_id.strip()
    
    file_path = None
    parser = None
    
    if "openhands" in agent:
        # Search pattern: logs/openhands/logs/issue_<TASK>/trajectory.json
        # The issue folder name usually contains the task_id or map to it
        # Actually, let's search recursively for json files containing the task_id
        search_path = os.path.join(logs_root, "openhands", "logs", "**", f"trajectory.json")
        candidates = glob.glob(search_path, recursive=True)
        # Filter for folder name containing task_id
        for c in candidates:
            if task_id in c: # Simple match
                file_path = c
                break
        
        # Fallback: Maybe the folder is named differently, search for task_id.json
        if not file_path:
             search_path = os.path.join(logs_root, "openhands", "logs", "**", f"{task_id}.json")
             candidates = glob.glob(search_path, recursive=True)
             if candidates: file_path = candidates[0]
             
        parser = parse_openhands_trajectory

    elif "live" in agent: # live-swe-agent
         search_path = os.path.join(logs_root, "live-swe-agent", "**", f"*.traj.json")
         candidates = glob.glob(search_path, recursive=True)
         for c in candidates:
             if task_id in os.path.basename(c):
                 file_path = c
                 break
         parser = parse_live_sweagent_trajectory

    elif "swe" in agent: # swe-agent
         # Search pattern: logs/swe-agent/issue_X/task_id.traj
         search_path = os.path.join(logs_root, "swe-agent", "**", f"*.traj")
         candidates = glob.glob(search_path, recursive=True)
         for c in candidates:
             if task_id in os.path.basename(c):
                 file_path = c
                 break
         parser = parse_sweagent_trajectory
         
    elif "meta" in agent: # metagpt
         # Search pattern: logs/metagpt/**/task_id.txt (recursive)
         search_path = os.path.join(logs_root, "metagpt", "**", f"*{task_id}*.txt")
         candidates = glob.glob(search_path, recursive=True)
         if not candidates:
             # Try non-recursive fallback
             search_path = os.path.join(logs_root, "metagpt", f"*{task_id}*.txt")
             candidates = glob.glob(search_path)
             
         if candidates: file_path = candidates[0]
         parser = parse_metagpt_trajectory
    
    return file_path, parser

def main():
    parser = argparse.ArgumentParser(description="View Agent Trajectory for Manual Annotation")
    parser.add_argument("--agent", required=True, help="Agent name (OpenHands, SWE-agent, MetaGPT, live-swe-agent)")
    parser.add_argument("--task", required=True, help="Task ID (e.g., scikit-learn__scikit-learn-12585)")
    parser.add_argument("--logs-dir", default="../logs", help="Path to logs directory")
    
    args = parser.parse_args()
    
    # Resolve logs dir relative to this script if default
    if args.logs_dir == "../logs":
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        logs_dir = os.path.join(base_dir, "logs")
    else:
        logs_dir = args.logs_dir

    print(f"Searching for log for Agent: {args.agent}, Task: {args.task} in {logs_dir}...")
    
    file_path, trace_parser = find_log_file(args.agent, args.task, logs_dir)
    
    if not file_path:
        print(f"{RED}Error: Could not find log file for {args.agent} and {args.task}{RESET}")
        return
        
    print(f"{GREEN}Found log: {file_path}{RESET}")
    print("Parsing...")
    
    try:
        if "openhands" in args.agent.lower():
            trace = trace_parser(file_path, "OpenHands")
        elif "live" in args.agent.lower():
             trace = trace_parser(file_path, "live-swe-agent")
        elif "swe" in args.agent.lower():
            # SWE-agent parser needs config path usually, but let's try just traj
            trace = trace_parser(file_path, "SWE-agent") 
        elif "meta" in args.agent.lower():
            trace = trace_parser(file_path, "MetaGPT")
            
        print(f"\n{BOLD}=== TRAJECTORY SUMMARY ==={RESET}")
        print(f"Agent: {trace.agent_name}")
        print(f"Task: {trace.task_id}")
        print(f"Steps: {len(trace.steps)}")
        print(f"Cost: ${trace.total_cost:.4f}")
        print("==========================\n")
        
        input("Press Enter to start reading the steps...")
        
        for step in trace.steps:
            print_step(step)
            # Optional: Pause every few steps? No, scrolling is better.
            
        print(f"\n{BOLD}=== END OF TRACE ==={RESET}")
        print("Now please fill the scores in manual_annotations.csv")
        
    except Exception as e:
        print(f"{RED}Error parsing trace: {e}{RESET}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
