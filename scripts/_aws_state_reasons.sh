# An EC2 instance's state reason, and what the P3/P6 driver does on it (the
# researcher's audit of 28 Nov 2026). Sourced, not executed:
#
#     source "$(dirname "${BASH_SOURCE[0]}")/_aws_state_reasons.sh"
#
# The reasons are the ones AWS documents for the StateReason object
# (https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_StateReason.html,
# read 28 Nov 2026). That page lists them under `message`, not `code`, and its
# troubleshooting page shows the two differing ("Message": "Client.VolumeLimitExceeded:
# Volume limit exceeded", "Code": "Server.InternalError";
# https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/troubleshooting-launch.html).
# So the reason is the message's leading token when it has one, else the code.
#
#   Server.SpotInstanceTermination    terminated by Spot for price or capacity
#   Server.SpotInstanceShutdown       stopped by Spot for price or capacity
#   Client.InstanceInitiatedShutdown  shut down from its own operating system:
#                                     the planned halt or the idle halt (a one-time
#                                     Spot instance terminates: "For one-time
#                                     requests, only Terminate is valid")
#   Client.UserInitiatedShutdown      shut down through the EC2 API (a terminate by hand)
#   Server.ScheduledStop              stopped for a scheduled retirement
#   Server.InsufficientInstanceCapacity, Server.InternalError, Client.InternalError,
#   Client.InvalidSnapshot.NotFound, Client.VolumeLimitExceeded
#                                     failed launches (the instance ends during launch)
#   Client.InstanceTerminated         terminated or rebooted during AMI creation
#   Client.UserInitiatedHibernate     hibernation started
#
# The driver's three paths, for an instance whose SSH is lost or whose pull or
# block start failed:
#   relaunch     a Spot instance shutting down or terminated by Spot or by its own
#                halt: the block in flight discarded, a relaunch under the limit
#   stop         an instance shutting down or terminated for any other reason, or
#                any reason on an on-demand instance: nothing can come back, so the
#                driver stops at once and reports the reason (no relaunch)
#   no-relaunch  an instance not terminated (stopped, stopping, running, pending,
#                or its state unreadable): the driver keeps polling to the block's
#                deadline; a stopped on-demand instance has the card's recovery

aws_state_reason() {   # aws_state_reason CODE MESSAGE: the documented reason
  local m="${2%%:*}"
  if [[ "$m" =~ ^(Server|Client)\.[A-Za-z.]+$ ]]; then echo "$m"
  elif [[ "$1" =~ ^(Server|Client)\.[A-Za-z.]+$ ]]; then echo "$1"
  else echo "none"
  fi
}

aws_state_reason_path() {   # aws_state_reason_path MARKET STATE REASON: relaunch | stop | no-relaunch
  case "$2" in
    shutting-down|terminated) ;;
    *) echo no-relaunch; return ;;
  esac
  if [[ "$1" == spot ]]; then
    case "$3" in
      Server.SpotInstanceTermination|Server.SpotInstanceShutdown|Client.InstanceInitiatedShutdown)
        echo relaunch; return ;;
    esac
  fi
  echo stop
}

aws_state_reason_said() {   # aws_state_reason_said REASON: the reason, in words
  case "$1" in
    Server.SpotInstanceTermination|Server.SpotInstanceShutdown) echo "$1: Spot reclaimed it" ;;
    Client.InstanceInitiatedShutdown) echo "$1: its own halt, taken as an interruption" ;;
    Client.UserInitiatedShutdown) echo "$1: terminated through the EC2 API" ;;
    none) echo "no state reason given" ;;
    *) echo "$1" ;;
  esac
}
