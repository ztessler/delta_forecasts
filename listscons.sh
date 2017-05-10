#!/bin/bash

ps aux | awk '/python.*[s]cons/ {print $0}'
