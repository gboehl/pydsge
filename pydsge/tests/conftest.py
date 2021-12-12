import os
import pytest

#from test_stability import mess11

"""
assuming that each test will fail, if a test passes, the corresponding messsage will be removed from the list of messages. 
"""
messages=  ["test 1 failed", "test 2 failed", "test 3 failed", "Test(s) failed, to update the pkl file, run ##########"] 

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
Todo: 
1. we can write general message about updating pkl which will be printed is any of the test fails. 
    problem with this: the error mesage will be printed for all tests, even for the tests that dont require pkl
2. To import varible 'messages' from test_stabilty, to conftest. and then print it if any test fails. 


any other ideas? 
"""