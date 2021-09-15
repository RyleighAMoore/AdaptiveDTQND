# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 18:44:41 2021

@author: Jacob
"""

# Go to the relevant when2meet page.
# Open the console. (ctr+shift+i)
# Run console.log(JSON.stringify([PeopleIDs, PeopleNames, AvailableAtSlot, TimeOfSlot]))
# Copy the output to data.json in the same folder as this script.
# The printed times are in your local timezone.

numberOfMeetings = 3
lengthOfMeetingInMinutes = 30
timeslotSizeInSeconds = 900

import json
from math import ceil
from copy import deepcopy
import time

with open('data.json') as fp:
    data_string = fp.read()
    data = json.loads(json.loads(data_string))

people = dict(zip(data[0], data[1]))
allTimes = dict(zip(data[3], data[2]))

lengthOfMeeting = ceil(lengthOfMeetingInMinutes / (timeslotSizeInSeconds / 60))

def getValidMeetings(lengthOfMeeting, times):
    validMeetings = {}
    for ix, meetingStartTime in enumerate(times):
        meetingEndTime = meetingStartTime + ( lengthOfMeeting -1) * timeslotSizeInSeconds
        if meetingEndTime in times:
            validMeetings[meetingStartTime] = set()
            for slot in range(lengthOfMeeting):
                people = times[meetingStartTime + slot * timeslotSizeInSeconds]
                validMeetings[meetingStartTime].update(people)
    return validMeetings

def getMeetingTimesFromMeetingId(meetingId):
    return [slot * timeslotSizeInSeconds + meetingId for slot in range(lengthOfMeeting)]

def getPeopleFromMeeting(meetingId, times):
    for meetingTime in getMeetingTimesFromMeetingId(meetingId):
        try:
            somePeople.intersection(set(times[meetingTime]))
        except:
            somePeople = set(times[meetingTime])
    return somePeople

def getTimesWithoutMeetingOrAttendees(meetingId, times):
    newTimes = deepcopy(times)
    peopleToRemove = getPeopleFromMeeting(meetingId, times)
    for time in newTimes:
        newTimes[time] = [x for x in newTimes[time] if x not in peopleToRemove]
    for meetingTime in getMeetingTimesFromMeetingId(meetingId):
        del newTimes[meetingTime]
    return newTimes

successfulAttendees = set()
times = deepcopy(allTimes)
for n in range(1, numberOfMeetings+1):
    validMeetings = getValidMeetings(int(lengthOfMeetingInMinutes/15), times)
    meetingWithMostAttendees = max(validMeetings, key=lambda meetingTime: len(validMeetings[meetingTime]))
    meetingWithMostAttendeesReadable = time.strftime('%b %d, %I:%M %p', time.gmtime(meetingWithMostAttendees))
    successfulAttendees.update(getPeopleFromMeeting(meetingWithMostAttendees, times))
    print(f'Meeting {n}: {meetingWithMostAttendeesReadable}, with {len(getPeopleFromMeeting(meetingWithMostAttendees, allTimes))} total attendees ({100 * len(getPeopleFromMeeting(meetingWithMostAttendees, allTimes))/len(people):.1f}%) and {len(getPeopleFromMeeting(meetingWithMostAttendees, times))} new attendees.')
    times = getTimesWithoutMeetingOrAttendees(meetingWithMostAttendees, times)

print(f'{len(successfulAttendees)} ({100 * len(successfulAttendees)/len(people):.1f}%) people can make it to at least one of the meetings.')
print(f'{len(people) - len(successfulAttendees)} ({100 * (len(people) - len(successfulAttendees))/len(people):.1f}%) people cannot attend one of the above meetings: {[people[peopleId] for peopleId in people if peopleId not in successfulAttendees]}')