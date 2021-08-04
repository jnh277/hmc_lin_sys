

inline var myfunc(const var& x_var, std::ostream* pstream)
{
    // compute value of function
    double x = x_var.val();
    double f = x * x;

    // compute partial derivatives
    double df_dx = 2.0 * x;

    // construct the autodiff wrapper
    return var(new precomp_vv_vari(
        f,          // the value of the output
        x_var.vi_,  // the input gradient wrt x
        df_dx       // the partial introduced by this function wrt x
    ));



}