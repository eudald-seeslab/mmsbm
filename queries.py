def get_cog(classroom):
    return f"""
    select profile_user.first_name, session_studentsession.student_id, session_studentsession.session_id, results_resultsitem.item_id, results_resultsitem.time as item_time, results_resultsitemsummary.correct as correct, results_resultsitemsummary.incorrect as incorrect 
    from profile_user, profile_student, results_resultsitem, results_resultsitemsummary, results_resultsactivity, session_studentsession, profile_studentclassroom
    where results_resultsitem.activity_result_id = results_resultsactivity.id and results_resultsactivity.student_session_id = session_studentsession.id and profile_student.id = session_studentsession.student_id and profile_student.user_id = profile_user.id and profile_studentclassroom.student_id=profile_student.id and profile_studentclassroom.classroom_id={classroom} and results_resultsitemsummary.result_id = results_resultsitem.id and results_resultsitem.activity_result_id = results_resultsactivity.id
    group by profile_user.first_name,  session_studentsession.student_id, session_studentsession.session_id, results_resultsactivity.id, results_resultsactivity.session_activity_id, results_resultsitem.item_id, item_time, correct, incorrect 
    order by session_studentsession.student_id, session_studentsession.session_id, results_resultsactivity.session_activity_id;
    """


def get_ada(classroom):
    return f"""
    select results_resultsselectraw.detail, session_studentsession.student_id as student, session_studentsession.session_id, results_resultsactivity.id as rra, results_resultsactivity.session_activity_id, results_resultsitem.item_id, results_resultsitem.type   
    from results_resultsselectraw,  profile_student, results_resultsitem, results_resultsactivity, session_studentsession, profile_studentclassroom 
    where results_resultsitem.activity_result_id = results_resultsactivity.id and results_resultsactivity.student_session_id = session_studentsession.id and results_resultsitem.id=results_resultsselectraw.result_id and profile_student.id = session_studentsession.student_id and profile_studentclassroom.student_id=profile_student.id and profile_studentclassroom.classroom_id={classroom} and session_studentsession.session_id=42
    """


def get_soc(classroom):
    return f"""
    select results_resultsselectraw.time, results_resultsselectraw.id, results_resultsselectraw.detail, results_resultsselectraw.action, session_studentsession.student_id as student, session_studentsession.session_id,   results_resultsitem.item_id, results_resultsactivity.session_activity_id 
    from results_resultsselectraw,  profile_student, results_resultsitem, results_resultsactivity, session_studentsession, profile_studentclassroom 
    where results_resultsitem.activity_result_id = results_resultsactivity.id and results_resultsactivity.student_session_id = session_studentsession.id and results_resultsitem.id=results_resultsselectraw.result_id and profile_student.id = session_studentsession.student_id and profile_studentclassroom.student_id=profile_student.id and profile_studentclassroom.classroom_id={classroom} and session_studentsession.session_id=30
    """


def get_gen(classroom):
    return f"""
    select results_resultsselectraw.id, results_resultsactivity.id as rra, results_resultsselectraw.detail, session_studentsession.student_id as student, session_studentsession.session_id,   results_resultsselectraw.action, results_resultsitem.item_id, results_resultsactivity.session_activity_id 
    from results_resultsitem, results_resultsactivity, session_studentsession, profile_studentclassroom, profile_student, results_resultsselectraw 
    where results_resultsitem.activity_result_id = results_resultsactivity.id and results_resultsactivity.student_session_id = session_studentsession.id and profile_student.id = session_studentsession.student_id  and profile_studentclassroom.student_id=profile_student.id and results_resultsitem.item_id= 1381 and results_resultsitem.id=results_resultsselectraw.result_id and profile_studentclassroom.classroom_id={classroom} and session_studentsession.session_id=26
    """


def get_act_items():
    return f"""
    select session_activity.id, session_activity.description, session_activity.type, session_activityitem.item_id
    from session_activity, session_activityitem
    where session_activity.id = session_activityitem.activity_id
    """
