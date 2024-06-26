from dm_control import suite
domainName = "walker"

tasks = [task_name for domain_name, task_name in suite.ALL_TASKS if domain_name == domainName]

print("Available tasks for  domain ",domainName," : ", tasks)