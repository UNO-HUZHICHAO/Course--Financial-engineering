import pexpect
import sys

password = "8bJ1KbTjj2kl"
host = "root@connect.bjb2.seetacloud.com"
port = "35998"
tar_file = "kcb50_project.tar.gz"

print("1. Uploading project to AutoDL...")
child = pexpect.spawn(f"scp -P {port} -o StrictHostKeyChecking=no {tar_file} {host}:/root/", encoding='utf-8')
try:
    i = child.expect(['password:', pexpect.EOF, pexpect.TIMEOUT], timeout=300)
    if i == 0:
        child.sendline(password)
        child.expect(pexpect.EOF, timeout=300)
        print("Upload complete!")
    else:
        print("Upload finished without password prompt or timeout.")
except Exception as e:
    print(f"Upload failed: {e}")
    sys.exit(1)

print("\n2. Connecting to AutoDL to start training in background...")
ssh_cmd = f"ssh -p {port} -o StrictHostKeyChecking=no {host}"
child = pexpect.spawn(ssh_cmd, encoding='utf-8')
child.logfile = sys.stdout

try:
    i = child.expect(['password:', pexpect.EOF, pexpect.TIMEOUT], timeout=30)
    if i == 0:
        child.sendline(password)
    
    child.expect([r'#', r'\$'], timeout=30)
    
    print("\n[Extracting files...]")
    child.sendline(f"tar -xzf {tar_file}")
    child.expect([r'#', r'\$'], timeout=60)
    
    print("\n[Submitting background task (setup + backtest)...]")
    # Use nohup to run setup and training so it survives network disconnections
    child.sendline("nohup bash -c 'bash setup_cloud.sh && python3 src/run_backtest.py' > run.log 2>&1 &")
    child.expect([r'#', r'\$'], timeout=10)
    
    # Give it a second to start and print the first few lines of the log
    child.sendline("sleep 2; head -n 15 run.log")
    child.expect([r'#', r'\$'], timeout=10)
    
    print("\n[Checking running processes...]")
    child.sendline("ps aux | grep run_backtest")
    child.expect([r'#', r'\$'], timeout=10)
    
    child.sendline("exit")
    child.expect(pexpect.EOF)
    print("\n[Success] Task is now running safely in the background on AutoDL!")
    
except Exception as e:
    print(f"SSH Error: {e}")
