#!/usr/bin/env python3
"""
Test runner script for Sidecar chat functionality
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description, cwd=None):
    """Run a command and handle output"""
    print(f"\n🔄 {description}...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print("Output:", result.stdout[-500:])  # Last 500 chars
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print("Error:", e.stderr)
        if e.stdout:
            print("Output:", e.stdout)
        return False


def install_test_dependencies():
    """Install additional test dependencies"""
    additional_deps = [
        "pytest-cov",
        "pytest-html", 
        "pytest-xdist",
        "pytest-timeout",
        "pytest-mock"
    ]
    
    for dep in additional_deps:
        cmd = [sys.executable, "-m", "pip", "install", dep]
        if not run_command(cmd, f"Installing {dep}"):
            print(f"⚠️  Failed to install {dep}, continuing anyway...")


def run_test_suite(test_type="all", verbose=False, coverage=False, parallel=False):
    """Run the test suite with specified options"""
    
    # Base pytest command
    cmd = [sys.executable, "-m", "pytest"]
    
    # Add verbosity
    if verbose:
        cmd.extend(["-v", "-s"])
    
    # Add coverage
    if coverage:
        cmd.extend([
            "--cov=app",
            "--cov-report=html:htmlcov",
            "--cov-report=term-missing"
        ])
    
    # Add parallel execution
    if parallel:
        cmd.extend(["-n", "auto"])
    
    # Select test type
    if test_type == "unit":
        cmd.extend(["-m", "unit"])
    elif test_type == "integration":
        cmd.extend(["-m", "integration"])
    elif test_type == "performance":
        cmd.extend(["-m", "performance"])
    elif test_type == "edge_case":
        cmd.extend(["-m", "edge_case"])
    elif test_type == "fast":
        cmd.extend(["-m", "not slow"])
    elif test_type == "slow":
        cmd.extend(["-m", "slow"])
    elif test_type == "api":
        cmd.extend(["-m", "api"])
    elif test_type == "search":
        cmd.extend(["-m", "search"])
    elif test_type == "conversation":
        cmd.extend(["-m", "conversation"])
    elif test_type == "model":
        cmd.extend(["-m", "model"])
    elif test_type == "gptj_validation":
        cmd.extend(["-m", "gptj_validation"])
    elif test_type == "environment":
        cmd.extend(["-m", "environment"])
    elif test_type == "download":
        cmd.extend(["-m", "download"])
    elif test_type != "all":
        cmd.append(f"tests/test_chat_{test_type}.py")
    
    return run_command(cmd, f"Running {test_type} tests")


def run_linting():
    """Run code linting"""
    linting_commands = [
        (["python", "-m", "flake8", "app/", "tests/"], "Flake8 linting"),
        (["python", "-m", "black", "--check", "app/", "tests/"], "Black formatting check"),
        (["python", "-m", "isort", "--check-only", "app/", "tests/"], "Import sorting check"),
    ]
    
    all_passed = True
    for cmd, desc in linting_commands:
        if not run_command(cmd, desc):
            all_passed = False
            print(f"⚠️  {desc} failed, continuing...")
    
    return all_passed


def generate_test_report():
    """Generate comprehensive test report"""
    print("\n📊 Generating Test Report...")
    
    # Run all tests with coverage
    cmd = [
        sys.executable, "-m", "pytest",
        "--cov=app",
        "--cov-report=html:htmlcov", 
        "--cov-report=xml",
        "--cov-report=term",
        "--html=test_report.html",
        "--self-contained-html",
        "--durations=20"
    ]
    
    if run_command(cmd, "Generating comprehensive test report"):
        print("✅ Test report generated:")
        print("   - Coverage HTML: htmlcov/index.html")
        print("   - Test report: test_report.html")
        print("   - Coverage XML: coverage.xml")
        return True
    return False


def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description="Sidecar Chat Test Runner")
    parser.add_argument(
        "test_type",
        nargs="?",
        default="all",
        choices=[
            "all", "unit", "integration", "performance", "edge_case",
            "fast", "slow", "api", "search", "conversation", "model",
            "gptj_validation", "environment", "download",
            "api_contracts", "edge_cases"
        ],
        help="Type of tests to run"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-c", "--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("-p", "--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--install-deps", action="store_true", help="Install test dependencies")
    parser.add_argument("--lint", action="store_true", help="Run linting checks")
    parser.add_argument("--report", action="store_true", help="Generate comprehensive report")
    parser.add_argument("--ci", action="store_true", help="CI mode (install deps, lint, test, report)")
    
    args = parser.parse_args()
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print("🧪 Sidecar Chat Test Suite")
    print("=" * 50)
    print(f"Working directory: {os.getcwd()}")
    print(f"Python version: {sys.version}")
    
    success = True
    
    # CI mode - run everything
    if args.ci:
        print("\n🚀 Running in CI mode...")
        
        if not install_test_dependencies():
            print("❌ Failed to install dependencies")
            success = False
        
        if not run_linting():
            print("❌ Linting checks failed")
            success = False
        
        if not run_test_suite("all", verbose=True, coverage=True, parallel=True):
            print("❌ Test suite failed")
            success = False
        
        if not generate_test_report():
            print("❌ Report generation failed")
            success = False
    
    else:
        # Individual options
        if args.install_deps:
            install_test_dependencies()
        
        if args.lint:
            if not run_linting():
                success = False
        
        if not run_test_suite(args.test_type, args.verbose, args.coverage, args.parallel):
            success = False
        
        if args.report:
            if not generate_test_report():
                success = False
    
    # Summary
    print("\n" + "=" * 50)
    if success:
        print("🎉 All operations completed successfully!")
        return 0
    else:
        print("💥 Some operations failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())