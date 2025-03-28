function ncu_trace {
    export CUDA_INJECTION64_PATH=none
    dyno dcgm_profiling --mute=true --duration=100000_s
    
    # Check if profiling is restricted to admin users
    if [ ! -w "/dev/nvidia0" ]; then
        echo "NVreg_RestrictProfilingToAdminUsers=1, running with sudo"
        SUDO="sudo"
    else
        SUDO=""
    fi

    # First generate the report file
    $SUDO ${CUDA_HOME}/bin/ncu --set full \
        --import-source yes \
        -o "$(date +%s)_profile" \
        -f \
        $1 "${@:2}"

    # Then convert to CSV with specific metrics
    ${CUDA_HOME}/bin/ncu -i "$(date +%s)_profile.ncu-rep" \
        --csv \
        --page raw \
        --metrics gpu__time_duration.sum,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
lts__t_sectors_srcunit_tex_op_read.sum,\
lts__t_sectors_srcunit_tex_op_write.sum,\
sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,\
smsp__inst_executed.sum \
        -o "$(date +%s)_metrics.csv"

    # Display the CSV contents
    cat "$(date +%s)_metrics.csv"
}