const STATUS_INITIAL = 0, STATUS_SAVING = 1, STATUS_SUCCESS = 2, STATUS_FAILED = 3;

window.onload = function (){
    new Vue({
        el: '#app_control',
        data: {
            title: 'jjjdlkjf',
            models :[],
            mode: 'start',
            selected_model:'',
            work_header:'',
            uploadedFiles: [],
            uploadError: null,
            currentStatus: null,
            uploadFieldName: 'photos',
            predict_result : {}
        },
        computed: {
          isInitial: function() {
            return this.currentStatus === STATUS_INITIAL;
          },
          isSaving: function() {
            return this.currentStatus === STATUS_SAVING;
          },
          isSuccess: function() {
            return this.currentStatus === STATUS_SUCCESS;
          },
          isFailed: function() {
            return this.currentStatus === STATUS_FAILED;
          }
        },
        mounted: function() {
            this.get_models();
            this.reset();
        },
        watch:{
            models:function(){},
            predict_result: function(value){
                this.predict_result = value;
                // console.log(value)
            },
        },
        methods:{
            get_models: function () {
                var vm = this; // Keep reference to viewmodel object
                $.get('/get_models', function(data){
                    vm.models = JSON.parse(data);
                });
            },
            show_details: function(model){
                this.mode = 'model_details';
                this.work_header = 'Model Details';
                this.selected_model = model;
                // this.models[this.selected_model].tp = 50;
                // this.models[this.selected_model].tn = 50;
                // this.models[this.selected_model].fn = 0;
                // this.models[this.selected_model].fp = 0;
            },
            create_model: function () {
                this.mode = 'create_model';
                this.work_header = 'Create New Model';
            },
            delete_model: function () {
                alert('delete!!!')
            },
            train_model: function () {
                alert('train!!!')
            },
            predict_model: function () {
                this.mode = 'model_predict';
                this.work_header = 'Predict Image';
            },
            post_create: function () {
                var model_name = this.$refs['model_name'].value;
                var img_rows = this.$refs['img_rows'].value;
                var img_cols = this.$refs['img_cols'].value;
                var epoch = this.$refs['epoch'].value;
                var kernel_size = this.$refs['kernel_size'].value;
                var pool_size = this.$refs['pool_size'].value;

                // TODO: add validation and show error msg.
                var vm = this;
                $.post('/add_model',{
                    'model_name': model_name,
                    'img_rows': img_rows,
                    'img_cols': img_cols,
                    'epoch': epoch,
                    'kernel_size': kernel_size,
                    'pool_size': pool_size,
                },function(data,status){
                    if(status=="success"){
                        models.append(model_name);
                        vm.$refs['model_name'].value = '';
                    }
                    else{
                        // TODO: show error msg
                    }
                })
            },
            reset: function() {
                // reset form to initial state
                this.currentStatus = STATUS_INITIAL;
                this.predict_result = {};
            },
            filesChange: function(fieldName, fileList) {
                // handle file changes
                var fd = new FormData();
                var vm = this; // Keep reference to viewmodel object

                var fileDict = {}
                if (!fileList.length) return;

                 // append the files to FormData
                for (x in Object.keys(fileList)){
                    fd.append(fileList[x].name, fileList[x]);
                    fileDict[fileList[x].name] = fileList[x];
                }
                fd.append('model_name', this.selected_model);

                function get_listener(i) {
                    load_lisinter = function (e) {
                        vm.$refs[i][0].src = e.target.result;
                    };
                    return load_lisinter
                }


                $.ajax({
                  url: '/predict_images',
                  data: fd,
                  processData: false,
                  contentType: false,
                  type: 'POST',
                  success: function(data){
                      for(i=0;i<Object.keys( data.predictions).length; i++){
                          var reader = new FileReader();
                          k = Object.keys( data.predictions)[i];
                          vm.predict_result[k] = {'result': data.predictions[k]};
                          reader.addEventListener('load',get_listener(k),false);
                          reader.readAsDataURL(fileDict[k]);
                      }
                      vm.currentStatus = STATUS_SUCCESS;
                      console.log(resultBuff);
                  }
                });
            },
            total_population: function () {
                var m = this.models[this.selected_model]
                return (m.tp + m.tn + m.fp + m.fn);
            },
            true_positive: function () {
                return this.models[this.selected_model].tp
            },
            true_negative: function () {
                return this.models[this.selected_model].tn
            },
            false_positive: function () {
                return this.models[this.selected_model].fp
            },
            false_negative: function () {
                return this.models[this.selected_model].fn
            },
            prevalence: function () {
                var tot_pos = this.true_negative() + this.false_positive();
                return ((tot_pos/this.total_population())*100).toFixed(2);
            },
            accuracy: function () {
                var tot_true = this.true_negative() + this.true_positive();
                return ((tot_true/this.total_population())*100).toFixed(2);
            },
            precision: function () {
                var tot_pos = this.true_positive() + this.false_positive();
                return ((this.true_positive()/tot_pos)*100).toFixed(2);
            },
            fdr: function () {
                var tot_pos = this.true_positive() + this.false_positive();
                return ((this.false_positive()/tot_pos)*100).toFixed(2);
            },
            false_omission_rate: function () {
                var tot_negative = this.false_negative() + this.true_negative();
                return ((this.false_negative()/tot_negative)*100).toFixed(2);
            },
            negative_predictive_value: function () {
                var tot_negative = this.false_negative() + this.true_negative();
                return ((this.true_negative()/tot_negative)*100).toFixed(2);
            },
            tpr: function(){
                var tot_con_pos = this.true_positive() + this.false_negative();
                return ((this.true_positive()/tot_con_pos)*100).toFixed(2);
            },
            fnr: function(){
                var tot_con_pos = this.true_positive() + this.false_negative();
                return ((this.false_negative()/tot_con_pos)*100).toFixed(2);
            },
            fpr: function () {
                var tot_con_neg = this.false_positive() + this.true_negative();
                return ((this.false_positive()/tot_con_neg)*100).toFixed(2);
            },
            tnr: function () {
                var tot_con_neg = this.false_positive() + this.true_negative();
                return ((this.true_negative()/tot_con_neg)*100).toFixed(2);
            },
            positive_likelihood_ratio: function () {
                return (this.tpr()/this.fpr()).toFixed(2);
            },
            negative_likelihood_ratio: function () {
                return (this.fnr()/this.tnr()).toFixed(2);
            },
            diagnostic_odds_ratio: function () {
                return (this.positive_likelihood_ratio()/this.negative_likelihood_ratio()).toFixed(2);
            },
            score: function () {
                // return (2*((this.tpr()*this.precision())/(this.tpr()+this.precision()))).toFixed(2);
                return ((2/((1/this.tpr())+(1/this.precision())))/100).toFixed(2);
            }
        }
    });
};