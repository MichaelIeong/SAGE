"""Testcases"""
import json
import os
import re
import sys
import time
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from io import StringIO
from typing import Any
from typing import Callable

import numpy as np
from dateutil import parser
from langchain.schema.messages import HumanMessage
from langchain.utilities import OpenWeatherMapAPIWrapper

from sage.misc_tools.gcloud_auth import gcloud_authenticate
from sage.misc_tools.google_suite import GoogleCalendarListEventsTool
from sage.testing.fake_requests import db
from sage.testing.testing_utils import listen
from sage.testing.testing_utils import manual_gmail_search
from sage.testing.testing_utils import pretty_print_email
from sage.testing.testing_utils import setup


tv_id = "8e20883f-c444-4edf-86bf-64e74c1d70e2"
frame_tv_id = "415228e4-456c-44b1-8602-70f98c67fba8"
tvs = [tv_id, frame_tv_id]
nightstand_light = "22e11820-64fa-4bed-ad8f-f98ef66de298"
fireplace_light = "571f0327-6507-4674-b9cf-fe68f7d9522d"
dining_table_light = "c4b4a179-6f75-48ed-a6b3-95080be6afc6"
tv_light = "363f714d-e4fe-4052-a78c-f8c97542e709"
lights = [nightstand_light, fireplace_light, dining_table_light, tv_light]
fridge = "51f02f33-4b43-11bf-2a6d-e7b5cf5be0ee"
dishwasher = "6c645f61-0b82-235f-e476-8a6afc0e73dc"
switch = "2214d9f0-4404-4bf6-a0d0-aaee20458d66"


TEST_REGISTER = {
    "device_resolution": set(),
    "personalization": set(),
    "persistence": set(),
    "intent_resolution": set(),
    "command_chaining": set(),
    "simple": set(),
    "test_set": set(),
    "human_interaction": set(),
    "google": set(),
}


def get_test_challenges(test_name: str):
    """
    Given a test name returns a list of challenges the test poses
    """
    challeges = []

    for key, setvals in TEST_REGISTER.items():
        if test_name in setvals:
            challeges.append(key)

    return challeges


def register(names: list[str]):
    "Add the desired testcases to the test register"

    def wrapper(f):
        for name in names:
            TEST_REGISTER[name].add(f)

        return f

    return wrapper


def get_tests(
    test_class_list: list[Callable], combination="intersection"
) -> list[Callable]:
    """Selects the testcases to run"""
    tests = TEST_REGISTER[test_class_list[0]]

    for test_class in test_class_list[1:]:
        if combination == "union":
            tests = tests.union(TEST_REGISTER[test_class])

        if combination == "intersection":
            tests = tests.intersection(TEST_REGISTER[test_class])

    return list(tests)


def check_status_on(device_state: dict[str, Any], device_id_list: list[str]) -> bool:
    """
    Returns True if any of the listed devices are on. Else False
    """

    for device in device_id_list:
        if device_state[device]["main"]["switch"]["switch"]["value"] == "on":
            return True

    return False


@register(["device_resolution"])
def turn_on_tv(device_state, config):
    device_state[tv_id]["main"]["switch"]["switch"]["value"] = "off"
    device_state[frame_tv_id]["main"]["switch"]["switch"]["value"] = "on"

    test_id, coordinator = setup(device_state, config.coordinator_config)
    user_command = "Amal: turn on the TV"
    coordinator.execute(user_command)
    device_state = db.get_device_state(test_id)
    assert (
        device_state[tv_id]["main"]["switch"]["switch"]["value"] == "on"
    ), "TV was not turned on"