#!/usr/bin/env python3
"""
Quick Results Checker
=====================

This script checks the current results from completed experiments in your grid search,
even while the full experiment is still running.

Usage:
    python quick_results_checker.py
"""

import os
import re
import sys
from pathlib import Path
import json
from collections import defaultdict, Counter
from datetime import datetime

class QuickResultsChecker:
    """Quick checker for immediate results analysis"""
    
    def __init__(self):
        self.workspace_dir = Path("/home/samuel_lozano/cooked")
        self.gridsearch_dir = self.workspace_dir / "gridsearch"
        self.results_dir = Path("/data/samuel_lozano/cooked/gridsearch/dtde_FIRST_gridsearch_2025-12-02_03-13-28")
        
    def check_results_directory_access(self):
        """Check if we can access the results directory"""
        try:
            if self.results_dir.exists():
                return True, "Results directory accessible"
            else:
                return False, f"Results directory not found: {self.results_dir}"
        except Exception as e:
            return False, f"Cannot access results directory: {e}"
    
    def extract_quick_metrics_from_log(self) -> dict:
        """Extract what we can from the main log file"""
        log_file = self.gridsearch_dir / "output_grid_search.log"
        
        if not log_file.exists():
            return {"error": "Main log file not found"}
            
        results = {
            'total_experiments': 0,
            'experiments_started': 0,
            'completed_configs': [],
            'failed_configs': [],
            'errors': []
        }
        
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Extract configuration details from start messages
            start_pattern = r'Starting: (exp\d+).*?Training params: ([^\\n]+).*?Architecture: ([^\\n]+).*?Starting experiment: ([^\\n]+)'
            start_matches = re.findall(start_pattern, content, re.DOTALL)
            
            for match in start_matches:
                exp_name, train_params, arch, full_name = match
                config = {
                    'exp_name': exp_name,
                    'full_name': full_name.strip(),
                    'training_params': train_params.strip(),
                    'architecture': arch.strip(),
                    'status': 'started'
                }
                results['completed_configs'].append(config)
                
            results['experiments_started'] = len(results['completed_configs'])
            
            # Look for completion indicators
            completion_pattern = r'(exp\d+).*?completed|finished|done'
            completion_matches = re.findall(completion_pattern, content, re.IGNORECASE)
            
            # Look for error patterns
            error_pattern = r'Error.*?:(.*?)\\n'
            error_matches = re.findall(error_pattern, content, re.IGNORECASE)
            results['errors'] = error_matches[:10]  # Keep first 10 errors
            
        except Exception as e:
            results['error'] = f"Error reading log file: {e}"
            
        return results
    
    def check_for_alternative_results(self):
        """Check for any results files that might be accessible"""
        possible_locations = [
            Path("/tmp/grid_search_results"),
            Path(os.path.expanduser("~/grid_search_results")),
            self.workspace_dir / "results",
            self.gridsearch_dir / "results"
        ]
        
        found_results = []
        for location in possible_locations:
            if location.exists():
                files = list(location.glob("*exp*"))
                if files:
                    found_results.append({
                        'location': str(location),
                        'files': len(files),
                        'sample_files': [f.name for f in files[:5]]
                    })
                    
        return found_results
    
    def analyze_hyperparameter_patterns(self, configs):
        """Analyze patterns in the started experiments"""
        analysis = {}
        
        # Extract hyperparameters from config names
        lr_pattern = r'lr([0-9.]+)'
        arch_pattern = r'(small|medium|large)'
        penalty_pattern = r'(low|high)'
        reward_pattern = r'(only_final|guided)'
        
        lrs, archs, penalties, rewards = [], [], [], []
        
        for config in configs:
            full_name = config.get('full_name', '')
            
            lr_match = re.search(lr_pattern, full_name)
            if lr_match:
                lrs.append(float(lr_match.group(1)))
                
            arch_match = re.search(arch_pattern, full_name)
            if arch_match:
                archs.append(arch_match.group(1))
                
            penalty_match = re.search(penalty_pattern, full_name)
            if penalty_match:
                penalties.append(penalty_match.group(1))
                
            reward_match = re.search(reward_pattern, full_name)
            if reward_match:
                rewards.append(reward_match.group(1))
        
        analysis = {
            'learning_rates': Counter(lrs),
            'architectures': Counter(archs),
            'penalty_levels': Counter(penalties),
            'reward_types': Counter(rewards),
            'total_configs': len(configs)
        }
        
        return analysis
    
    def generate_report(self):
        """Generate a comprehensive quick report"""
        print("⚡ QUICK RESULTS CHECK")
        print("=" * 50)
        
        # Check results directory access
        can_access, access_msg = self.check_results_directory_access()
        print(f"🗂️ Results Directory Access: {access_msg}")
        
        # Analyze log file
        log_results = self.extract_quick_metrics_from_log()
        
        if "error" in log_results:
            print(f"❌ Log Analysis Error: {log_results['error']}")
            return
            
        print(f"\n📊 EXPERIMENT STATUS:")
        print(f"   • Experiments started: {log_results['experiments_started']}")
        print(f"   • Configurations tracked: {len(log_results['completed_configs'])}")
        
        if log_results['errors']:
            print(f"   • Errors detected: {len(log_results['errors'])}")
            print(f"   • Sample error: {log_results['errors'][0][:100]}...")
            
        # Analyze hyperparameter patterns
        if log_results['completed_configs']:
            analysis = self.analyze_hyperparameter_patterns(log_results['completed_configs'])
            
            print(f"\n🎯 HYPERPARAMETER ANALYSIS:")
            print(f"   • Learning Rates tested:")
            for lr, count in analysis['learning_rates'].most_common():
                print(f"     - {lr}: {count} experiments")
                
            print(f"   • Architecture sizes:")
            for arch, count in analysis['architectures'].most_common():
                print(f"     - {arch}: {count} experiments")
                
            print(f"   • Penalty configurations:")
            for penalty, count in analysis['penalty_levels'].most_common():
                print(f"     - {penalty}: {count} experiments")
                
            print(f"   • Reward configurations:")
            for reward, count in analysis['reward_types'].most_common():
                print(f"     - {reward}: {count} experiments")
                
        # Check for alternative results
        alt_results = self.check_for_alternative_results()
        if alt_results:
            print(f"\n📁 ALTERNATIVE RESULTS FOUND:")
            for result in alt_results:
                print(f"   • {result['location']}: {result['files']} files")
                
        # Provide next steps
        print(f"\n💡 NEXT STEPS:")
        if can_access:
            print(f"   ✅ Use: python fast_results_analyzer.py {self.results_dir}")
            print(f"   ✅ This will analyze completed experiments with performance metrics")
        else:
            print(f"   📋 Wait for more experiments to complete")
            print(f"   📋 Check system status and available storage")
            print(f"   📋 Consider running: python local_analyzer.py for status updates")
            
        print(f"\n🔄 MONITORING:")
        print(f"   • Re-run this script periodically for updates")
        print(f"   • Watch the main log file: {self.gridsearch_dir / 'output_grid_search.log'}")
        
        # Save quick analysis
        report_file = self.gridsearch_dir / "quick_results_report.json"
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'can_access_results': can_access,
            'log_analysis': log_results,
            'hyperparameter_analysis': analysis if 'analysis' in locals() else {},
            'alternative_results': alt_results
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        print(f"\n💾 Report saved to: {report_file}")


def main():
    """Main function"""
    checker = QuickResultsChecker()
    checker.generate_report()


if __name__ == "__main__":
    main()