ps -e -o psr,pid,%cpu,%mem,args | grep -E '^(PSR|[[:space:]]*10)'
