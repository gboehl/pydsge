import os
import pytest

#from test_stability import mess11

"""
assuming that each test will fail, if a test passes, the corresponding messsage will be removed from the list of messages. 
"""
messages=  ["test 1 failed", "test 2 failed", "test 3 failed", "Test(s) failed: To update the pkl file, run ##########"] 

def test_1():
    assert False
    messages.remove("test 1 failed")

def test_2():
    assert True
    messages.remove("test 2 failed")

def test_3():
    assert False
    messages.remove("test 3 failed")
        

me11= messages
"""
In the overview section in ouput, it shows which tests failed. if all test pass, there will be no overview section in the terminal. 
"""
def pytest_report_teststatus(report, config):
    if report.outcome != 'passed':
        line = f'{report.nodeid} says:\t"{me11}"'
        report.sections.append(('Test(s) Overview', line))
        

"""
to print custom error message for a specific test if it fails:


import logging

log = logging.getLogger('')
log.error('\n\n test_what_output_is_there Failed!! \n.')
assert False


"""
