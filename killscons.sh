#!/bin/bash

ps aux | awk '/python.*[s]cons/ {print $2}' | xargs -L 1 kill -9
