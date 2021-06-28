#pragma once
#include <chrono>

#define __DoTestTime

#ifdef __DoTestTime
#define TestTimeVar std::chrono::steady_clock::time_point testTime_t1, testTime_t2;
#define TestTimeVarSum(x) double x = 0;

#define TestTimeTic testTime_t1 = std::chrono::steady_clock::now();

#define TestTimeToc(str)                            \
    testTime_t2 = std::chrono::steady_clock::now(); \
    printf("%s: %.1f ms\n", str,                    \
           1000 * std::chrono::duration_cast<std::chrono::duration<double>>(testTime_t2 - testTime_t1).count());

#define TestTimeTocFPS(str)                         \
    testTime_t2 = std::chrono::steady_clock::now(); \
    printf("%s FPS: %.1f\n", str,                   \
           1.0 / std::chrono::duration_cast<std::chrono::duration<double>>(testTime_t2 - testTime_t1).count());

#define TestTimeTocSum(x)                           \
    testTime_t2 = std::chrono::steady_clock::now(); \
    x += 1000 * std::chrono::duration_cast<std::chrono::duration<double>>(testTime_t2 - testTime_t1).count();

#define TestTimePrintf(str, x) printf("%s: %.1f\n", str, x);

#else
#define TestTimeVar
#define TestTimeVarSum(x)
#define TestTimeTic

#define TestTimeToc(str)
#define TestTimeTocSum(x)
#define TestTimePrintf(str, x)
#endif
