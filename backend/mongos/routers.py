class TestRouter:
    """
    mongotest 앱의 모델에서 수행되는 모든 데이터베이스 연산을 제어하는 중계기
    """

    def db_for_read(self, model, **hints):
        """
        mongotest 앱의 모델을 조회하는 경우 mongodb로 중계한다.
        """
        if model._meta.app_label == 'mongos':
            return 'mongodb'
        return None

    def db_for_write(self, model, **hints):
        """
        mongos 앱의 모델을 기록하는 경우 mongodb로 중계한다.
        """
        if model._meta.app_label == 'mongos':
            return 'mongodb'
        return None

    def allow_relation(self, obj1, obj2, **hints):
        """
        mongos 앱의 모델과 관련된 관계 접근을 허용한다.
        """
        if obj1._meta.app_label == 'mongos' or \
                obj2._meta.app_label == 'mongos':
            return True
        return None

    def allow_migrate(self, db, app_label, model_name=None, **hints):
        """
        mongos 앱의 모델에 대응하는 표가 mongodb 데이터베이스에만 생성되도록 한다.
        """
        if app_label == 'mongos':
            return db == 'mongodb'
        return None
