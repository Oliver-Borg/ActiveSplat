#!/bin/bash
# Launch the results dashboard in its own env. PYTHONPATH is cleared because sourcing the
# catkin workspace points it at the ActiveSplat (python3.9) site-packages.
ShellScriptFolder=$(cd $(dirname "$0"); pwd)
DashboardEnv=${DASHBOARD_ENV:-activesplat-dash}

exec conda run --no-capture-output -n $DashboardEnv env -u PYTHONPATH python $ShellScriptFolder/dashboard.py "$@"
