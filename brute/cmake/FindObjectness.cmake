# - Try to find Objectness
# Once done this will define
#  OBJECTNESS_FOUND - System has Objectness
#  OBJECTNESS_INCLUDE_DIRS - The Objectness include directories
#  OBJECTNESS_LIBRARIES - The libraries needed to use LibXml2

find_path(OBJECTNESS_INCLUDE_DIR Objectness/Objectness.h
	PATHS ${CMAKE_CURRENT_SOURCE_DIR}/include
)

find_library(OBJECTNESS_LIBRARY NAMES objectness
	PATHS ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

set(OBJECTNESS_LIBRARIES ${OBJECTNESS_LIBRARY} )
set(OBJECTNESS_INCLUDE_DIRS ${OBJECTNESS_INCLUDE_DIR} )

if (OBJECTNESS_INCLUDE_DIR AND OBJECTNESS_LIBRARIES)
	set(OBJECTNESS_FOUND true)
else()
	set(OBJECTNESS_FOUND false)
endif()

mark_as_advanced(OBJECTNESS_FOUND)
