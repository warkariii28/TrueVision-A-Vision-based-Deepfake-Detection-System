@app.route("/login/", methods=("GET", "POST"), strict_slashes=False)
def login():
    form = login_form()

    if form.validate_on_submit():
        try:
            user = User.query.filter_by(email=form.email.data).first()

            if user and bcrypt.check_password_hash(user.pwd, form.pwd.data):
                login_user(user)
                session.permanent = True
                next_page = request.args.get('next', url_for('upload'))
                return redirect(next_page)
            else:
                flash("Invalid username or password!", "danger")
        except Exception as e:
            flash(str(e), "danger")

    return render_template("auth.html",
        form=form,
        text="Login",
        title="Login",
        btn_action="Login",
    )
