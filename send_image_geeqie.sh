#!/bin/bash

IMAGESRC="/home/flappy/Embedded-Projects/temp_stream/frame.jpg"
IMAGEDEST="/home/observer1/Desktop/frame.jpg"
cp "$IMAGESRC" "$IMAGEDEST"

# Detect XRDP display
ps -C Xorg -o user=,uid=,args= | grep -v flappy | grep -v root |
	while read name uids skip dispnum rest; do
	
	touname=$(id -nu "$uids")
	DISPLAY=$(echo "$dispnum")
	DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$uids/bus"
	sudo -u "$touname" geeqie --remote "$IMAGEDEST" &
done
