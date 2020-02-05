#!/usr/bin/env python3
__author__ = 'Larry A. Hartman'
__company__ = 'Janus Research'

# Initialize Application processes
if __name__ == '__main__':
    from crontab import CronTab

    cron_sched = CronTab(user='pi')
    cron_sched.remove_all()
    cron_sched.write()

    job_stepper = cron_sched.new(command='sudo python3 /opt/Janus/WM/python3/main-stepper.py')
    # job_stepper.minute.on(0,15,30,45)
    job_stepper.minute.every(5)
    cron_sched.write()
