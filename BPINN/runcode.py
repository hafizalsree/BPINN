import subprocess
import time

# List of scripts to run
scripts = ['testrun.py', 'testrun.py', 'testrun.py']

start_time = time.time()

# Run each script and show progress bar
for i, script in enumerate(scripts):
    print(f'Running script {i+1}/{len(scripts)} ---> {script}')
    subprocess.run(['python', script])
    # show progress bar
    print(f'{"="*10} {i+1}/{len(scripts)} {"="*10}')

print('All scripts have been run')

# Calculate the total time taken
total_time = time.time() - start_time
print(f'Total time taken: {total_time:.2f} seconds')
