# References:
# https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
# and
# https://discourse.julialang.org/t/tcn-temporal-convolutional-networks-in-julia/46188

function tcn(in_out::Pair{Int,Int};
             channels::AbstractVector{Int}=[], kernel_size=3, dropout=0.2)
    from = vcat(first(in_out), channels)
    to = vcat(channels, last(in_out))

    return Chain((tcn_block(a => b; dilation=2^(i - 1), kernel_size, dropout)
                  for (i, (a, b)) in enumerate(zip(from, to)))...)
end

function tcn_block(in_out::Pair{Int,Int};
                   dilation::Int,
                   kernel_size=3,
                   dropout=0.2)
    # Note: n the original implementation the authors use "Chomping". Chomping
    # with symmetric padding is equivalent to asymmetric padding with no
    # chomping. The latter is simpler and requires less allocations.
    _in, out = in_out
    pad = ((kernel_size - 1) * dilation, 0)
    main = Chain(Conv((kernel_size,), in_out; dilation, pad),
                 BatchNorm(out),
                 PReLU(),
                 Dropout(dropout),
                 Conv((kernel_size,), out => out; dilation, pad),
                 BatchNorm(out),
                 PReLU(),
                 Dropout(dropout))
    if _in == out
        return Chain(SkipConnection(main, +),
                     PReLU())
    end
    shortcut = Conv((1,), in_out)
    return Chain(Parallel(+, main, shortcut),
                 PReLU())
end