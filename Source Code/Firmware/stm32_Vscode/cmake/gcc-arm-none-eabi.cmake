set(CMAKE_SYSTEM_NAME               Generic)
set(CMAKE_SYSTEM_PROCESSOR          arm)

set(CMAKE_C_COMPILER_ID GNU)
set(CMAKE_CXX_COMPILER_ID GNU)

# Discover the GNU ARM toolchain from the STM32CubeIDE bundle or a standard installation.
set(_arm_toolchain_hints)
if(DEFINED ENV{CUBE_BUNDLE_PATH})
    list(APPEND _arm_toolchain_hints "$ENV{CUBE_BUNDLE_PATH}")
    list(APPEND _arm_toolchain_hints "$ENV{CUBE_BUNDLE_PATH}/gnu-tools-for-stm32")
endif()
list(APPEND _arm_toolchain_hints
    "C:/ST/STM32CubeIDE_2.1.1/STM32CubeIDE/plugins"
    "C:/ST/STM32CubeIDE_1.18.1/STM32CubeIDE/plugins"
    "C:/Program Files/STMicroelectronics/STM32CubeIDE/plugins"
    "C:/Program Files (x86)/STMicroelectronics/STM32CubeIDE/plugins"
    "C:/TouchGFX/4.26.1/env/MinGW/msys/1.0/gnu-arm-gcc/bin"
)

set(_arm_toolchain_bin_dirs)
foreach(_hint IN LISTS _arm_toolchain_hints)
    if(EXISTS "${_hint}")
        file(GLOB _hint_matches "${_hint}/**/tools/bin/arm-none-eabi-gcc.exe")
        foreach(_match IN LISTS _hint_matches)
            get_filename_component(_bin_dir "${_match}" DIRECTORY)
            list(APPEND _arm_toolchain_bin_dirs "${_bin_dir}")
        endforeach()
        if(EXISTS "${_hint}/bin")
            list(APPEND _arm_toolchain_bin_dirs "${_hint}/bin")
        endif()
    endif()
endforeach()
list(REMOVE_DUPLICATES _arm_toolchain_bin_dirs)

find_program(ARM_NONE_EABI_GCC NAMES arm-none-eabi-gcc arm-none-eabi-gcc.exe
    HINTS ${_arm_toolchain_bin_dirs}
    REQUIRED)

get_filename_component(ARM_TOOLCHAIN_DIR "${ARM_NONE_EABI_GCC}" DIRECTORY)
set(ARM_TOOLCHAIN_PREFIX "${ARM_TOOLCHAIN_DIR}/arm-none-eabi-")

set(CMAKE_C_COMPILER                "${ARM_NONE_EABI_GCC}")
set(CMAKE_ASM_COMPILER              "${ARM_NONE_EABI_GCC}")
set(CMAKE_CXX_COMPILER              "${ARM_TOOLCHAIN_PREFIX}g++")
set(CMAKE_LINKER                    "${ARM_TOOLCHAIN_PREFIX}g++")
set(CMAKE_OBJCOPY                   "${ARM_TOOLCHAIN_PREFIX}objcopy")
set(CMAKE_SIZE                      "${ARM_TOOLCHAIN_PREFIX}size")

set(CMAKE_EXECUTABLE_SUFFIX_ASM     ".elf")
set(CMAKE_EXECUTABLE_SUFFIX_C       ".elf")
set(CMAKE_EXECUTABLE_SUFFIX_CXX     ".elf")

set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# MCU specific flags
set(TARGET_FLAGS "-mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard ")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${TARGET_FLAGS}")
set(CMAKE_ASM_FLAGS "${CMAKE_C_FLAGS} -x assembler-with-cpp -MMD -MP")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -fdata-sections -ffunction-sections -fstack-usage")

# The cyclomatic-complexity parameter must be defined for the Cyclomatic complexity feature in STM32CubeIDE to work.
# However, most GCC toolchains do not support this option, which causes a compilation error; for this reason, the feature is disabled by default.
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fcyclomatic-complexity")

set(CMAKE_C_FLAGS_DEBUG "-O0 -g3")
set(CMAKE_C_FLAGS_RELEASE "-Os -g0")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3")
set(CMAKE_CXX_FLAGS_RELEASE "-Os -g0")

set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -fno-rtti -fno-exceptions -fno-threadsafe-statics")

set(CMAKE_EXE_LINKER_FLAGS "${TARGET_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -T \"${CMAKE_SOURCE_DIR}/STM32F405XX_FLASH.ld\"")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --specs=nano.specs")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-Map=${CMAKE_PROJECT_NAME}.map -Wl,--gc-sections")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--print-memory-usage")
set(TOOLCHAIN_LINK_LIBRARIES "m")
