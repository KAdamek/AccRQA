# Define list of folders containing test files.
set(${CMAKE_PROJECT_NAME}_APP
    RQA
)

# Build tests.
foreach (applname ${${CMAKE_PROJECT_NAME}_APP})
    add_executable("${applname}.exe" "${applname}.cpp")
    target_link_libraries("${applname}.exe" ${CMAKE_PROJECT_NAME})
endforeach()
