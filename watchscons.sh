#!/bin/bash

watch -n 1 "ps aux | awk '/python.*[s]cons/ {print \$2}' | sort"
