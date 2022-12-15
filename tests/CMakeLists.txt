# Define list of folders containing test files.
set(${CMAKE_PROJECT_NAME}_TEST
    test_RQA
)

# Build tests.
foreach (test ${${CMAKE_PROJECT_NAME}_TEST})
    add_executable(${test} "${test}.cpp")
    target_link_libraries(${test} ${CMAKE_PROJECT_NAME})
    add_test(${test} ${test})
endforeach()
