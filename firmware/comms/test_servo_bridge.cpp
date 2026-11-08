#include "combined_only/servo_bridge.h"
#include <cassert>
#include <cstdio>

struct TraceBackend {
  int count = 0;
  int joint[4] = {};
  int angle[4] = {};
  void write(int id, int value) { joint[count]=id; angle[count++]=value; }
  void detachAll() {}
};

int main() {
  const int16_t lower[4] = {3000, 6000, 4500, 7000};
  const int16_t upper[4] = {15000, 12000, 13500, 11000};
  for (int joint=0; joint<4; ++joint) {
    for (int side=0; side<2; ++side) {
      ServoBridge<MockServoBackend> bridge;
      int16_t targets[4] = {9100, 8600, 10100, 9100};
      targets[joint] = side ? upper[joint]+1 : lower[joint]-1;
      assert(!bridge.apply(targets));
      assert(bridge.backend.writes == 0);
      for(int i=0;i<4;++i) assert(bridge.positions[i] == 90);
    }
  }
  ServoBridge<TraceBackend> trace;
  const int16_t targets[4] = {9149, 8550, 10051, 9000};
  const int expected[4] = {91, 86, 101, 90};
  assert(trace.apply(targets));
  assert(trace.backend.count == 4);
  for(int i=0;i<4;++i) {
    assert(trace.backend.joint[i] == i);
    assert(trace.backend.angle[i] == expected[i]);
    assert(trace.positions[i] == expected[i]);
  }
  ServoBridge<MockServoBackend> boundary;
  assert(boundary.apply(lower));
  assert(boundary.apply(upper));
  for(int mask=0; mask<16; ++mask) {
    for(int i=0;i<4;++i) boundary.backend.attached[i] = (mask >> i) & 1;
    boundary.timeout();
    for(int i=0;i<4;++i) {
      assert(boundary.backend.attached[i] == bool((mask >> i) & 1));
      assert(boundary.positions[i] == upper[i]/100);
    }
    boundary.stop();
    boundary.timeout();
    for(int i=0;i<4;++i) assert(!boundary.backend.attached[i]);
  }
  puts("PASS: 3B bridge order, rounding, limits, zero partial writes, STOP detach, timeout hold");
}
