#coding: utf-8

import os
from fabric.api import task, local, run, env, cd
from fabric.contrib.project import rsync_project

env.use_ssh_config = True
env.shell = "/bin/zsh -l -i -c"

local_dir = os.path.dirname(__file__)
remote_dir = "~/proj/"+os.path.dirname(__file__).split("/")[-1]

def background_run(cmd):
  run("sh -c '((nohup %s > /dev/null 2> /dev/null) & )'" % cmd, pty=False)

@task
def deploy(sync_input=False, sync_working=False, delete=False):
  run("mkdir -p %s/" % remote_dir)
  exclude = [".DS_Store", "*tmp*", "*.pyc", "data/output/"]
  if not sync_input:
    exclude.append("data/input/")
  if not sync_working:
    exclude.append("data/working/")
  rsync_project(
      local_dir=local_dir+"/",
      remote_dir=remote_dir+"/",
      exclude=exclude,
      delete=delete
  )

  with cd(remote_dir):
    # Register tasks here
    # background_run("python ./code/script/hello.py")
    pass

@task
def fetch(sync_working=True, sync_output=True):
  if sync_working:
    local("mkdir -p %s/data/working/" % local_dir)
    local("rsync -auvz %s:%s/data/working/ %s/data/working/" % (env.hosts[0], remote_dir, local_dir))
  if sync_output:
    local("mkdir -p %s/data/output/" % local_dir)
    local("rsync -auvz %s:%s/data/output/ %s/data/output/" % (env.hosts[0], remote_dir, local_dir))
