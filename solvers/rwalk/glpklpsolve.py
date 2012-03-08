import glpkpi as g

NORMAL=1
DETAILED=2
IMPORTANT=3
FULL=4

UNBOUNDED=-4 
INFEASIBLE=-3
SUBOPTIMAL=-2
NOMEMORY=-1 
OPTIMAL=0
DEGENERATE=2 
NUMFAILURE=3 
USERABORT=4 
TIMEOUT=5 
PRESOLVED=6


EQ=0
GE=1
LE=-1

def lpsolve(f, *args):
    if f == 'make_lp':
        nr,nc = args
        lp = g.glp_create_prob()
        if nr > 0: g.glp_add_rows(lp, nr)
        if nc > 0: g.glp_add_cols(lp, nc)

        for i in range(nc):
            g.glp_set_col_bnds(lp, i+1, g.GLP_LO, 0, 0)

        lp.glpk_parms = g.glp_smcp()
        g.glp_init_smcp(lp.glpk_parms)

        return lp

    if f == 'set_verbose':
        lp = args[0]
        lvl = args[1]
        if lvl == FULL:      lp.glpk_parms.msg_lev = g.GLP_MSG_ALL
        if lvl == DETAILED:  lp.glpk_parms.msg_lev = g.GLP_MSG_ALL
        if lvl == NORMAL:    lp.glpk_parms.msg_lev = g.GLP_MSG_ON
        if lvl == IMPORTANT: lp.glpk_parms.msg_lev = g.GLP_MSG_ERR

    if f == 'add_constraint':
        lp = args[0]
        args = args[1:]
        g.glp_add_rows(lp, 1)
        nr = g.glp_get_num_rows(lp)
        nc = g.glp_get_num_cols(lp)
        #print 'lp has %i %i' % (nr, nc)
        a = g.doubleArray(nc+1)
        b = g.intArray(nc+1)
        for i,v in enumerate(args[0]):
            a[i+1] = v
            b[i+1] = i+1

        assert i == nc-1

        if args[1] == EQ: g.glp_set_row_bnds(lp, nr, g.GLP_FX, args[2], 0)
        if args[1] == LE: g.glp_set_row_bnds(lp, nr, g.GLP_UP, 0,args[2])
        if args[1] == GE: g.glp_set_row_bnds(lp, nr, g.GLP_LO, args[2], 0)

        g.glp_set_mat_row(lp,nr, nc, b, a)

    if f == 'lp_solve_version':
        return 'glpk %s' % (g.glp_version(),)

    if 1 and f == 'solve':
        lp = args[0]
        r = g.glp_simplex(lp, lp.glpk_parms)
        if r == 0:
            r = g.glp_get_status(lp)
            if r == g.GLP_OPT: return OPTIMAL
            if r == g.GLP_FEAS: return SUBOPTIMAL
            if r == g.GLP_INFEAS: return INFEASIBLE
            if r == g.GLP_NOFEAS: return INFEASIBLE
            if r == g.GLP_UNBND: return UNBOUNDED
            if r == g.GLP_UNDEF: return DEGENERATE
        else:
            return r

    if 0 and f == 'solve':
        lp = args[0]
        #parms = g.glp_init_smcp(lp)
        #print parms
        #assert 0
        r = g.glp_interior(lp, None)
        if r == 0:
            r = g.glp_ipt_status(lp)
            if r == g.GLP_OPT: return OPTIMAL
            if r == g.GLP_INFEAS: return INFEASIBLE
            if r == g.GLP_NOFEAS: return INFEASIBLE
            if r == g.GLP_UNDEF: return DEGENERATE
        elif r == g.GLP_EITLIM: return OPTIMAL
        else:
            return r

    if f == 'get_total_iter':
        return 0

    if f == 'get_variables':
        lp = args[0]
        v = []
        for i in range(g.glp_get_num_cols(lp)):
            v.append(g.glp_get_col_prim(lp, i+1))
        return [v]

    if f == 'get_Norig_columns':
        return g.glp_get_num_cols(args[0])

    if f == 'get_Ncolumns':
        return g.glp_get_num_cols(args[0])

    if f == 'get_objective':
        return g.glp_get_obj_val(args[0])

    if f == 'set_sense':
        lp = args[0]
        if args[1]:
            g.glp_set_obj_dir(lp, g.GLP_MIN)
        else:
            g.glp_set_obj_dir(lp, g.GLP_MAX)
       


    if f == 'set_obj_fn':
        lp = args[0]
        for i,v in enumerate(args[1]):
            g.glp_set_obj_coef(lp, i+1, v)

