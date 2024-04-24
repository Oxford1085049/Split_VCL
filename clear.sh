#!/bin/bash

if [ -d "./plots/" ]; then
    rm -rf ./plots/*
    echo "Directory ./plots/ emptied."
fi

if [ -d "./outputs/" ]; then
    rm -rf ./outputs/*
    echo "Directory ./outputs/ emptied."
fi