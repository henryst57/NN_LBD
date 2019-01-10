#kills all python and perl scripts run by the user
#$1 is the username running the python and perl scripts

pkill -9 -u $1 python
pkill -9 -u $1 perl
