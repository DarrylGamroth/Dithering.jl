# Dithering.jl


```julia
using Dithering

const T = Float32

# Create dither object, 468 actuators, 12 bits, range [-1, 1], step size 0.2
d = Dither(T, 468, 12, -1, 1, 0.2)

# Set the target for the current input
target!(d, rand(T, 468))

# Perform an iteration and collect the output
signal = step!(d)
```

